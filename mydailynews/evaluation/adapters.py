from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Dict, List, Protocol

from mydailynews.ai.base import AIClient, AIJsonError, AITransportError
from mydailynews.analysis.delta import DeltaExtractor
from mydailynews.analysis.deterministic_delta import assess_lexical_change
from mydailynews.app.models import (
    DeltaExtractionConfig,
    HeadlineDecision,
    MemoryAnnotation,
    PriorReport,
    SelectedArticle,
    TopicConfig,
)
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.domain.candidate_annotations import set_memory_annotation
from mydailynews.domain.headline_selection import profile_match_signals
from mydailynews.domain.text_similarity import compare_token_sets
from mydailynews.evaluation.providers import FixtureNewsProvider
from mydailynews.evaluation.schema import (
    EvalArcInput,
    EvalCorpus,
    EvalPrediction,
)
from mydailynews.memory.story_index import MATCH_CONFIDENCE_THRESHOLD
from mydailynews.memory.story_keys import StoryIdentity, story_identity_for_candidate


class EvaluationAdapter(Protocol):
    name: str

    def predict(self, arc: EvalArcInput) -> List[EvalPrediction]: ...


@dataclass
class _ObservedStory:
    predicted_story_id: str
    identity: StoryIdentity
    title: str
    last_seen: str = ""


class ProductionHeuristicAdapter:
    """Runs the repository's deterministic identity/profile/delta fallback.

    It is intentionally not an oracle and does not pretend to measure final-text
    faithfulness. Its results are a baseline for what the non-LLM policy layer can
    and cannot establish.
    """

    def __init__(self, *, fixture_mode: str = "direct") -> None:
        self.fixture_mode = _fixture_mode(fixture_mode)
        self.name = f"production_heuristic:{self.fixture_mode}"

    def predict(self, arc: EvalArcInput) -> List[EvalPrediction]:
        provider = FixtureNewsProvider(arc)
        stories: Dict[str, _ObservedStory] = {}
        predictions: List[EvalPrediction] = []
        for day in arc.days:
            fetched = _fetch_fixture(provider, day.date, self.fixture_mode)
            for candidate in fetched.candidates:
                started = perf_counter()
                identity = story_identity_for_candidate(candidate)
                best_score, best_story = _best_observed_story(identity, stories)
                same_story = best_story is not None and best_score >= MATCH_CONFIDENCE_THRESHOLD
                if same_story and best_story is not None:
                    predicted_story_id = best_story.predicted_story_id
                else:
                    predicted_story_id = _unused_story_id(identity.story_key, stories)

                prior_title = best_story.title if best_story is not None and best_score >= 0.34 else None
                assessment = assess_lexical_change(
                    candidate.title,
                    prior_title,
                    story_key_match=same_story,
                )
                relevance = _profile_relevance(candidate, arc)
                display = assessment.disposition if assessment.disposition != "uncertain" else "full_report"
                selected = relevance != "irrelevant" and display != "omit"
                predictions.append(
                    EvalPrediction(
                        arc_id=arc.id,
                        date=day.date,
                        document_id=candidate.id,
                        predicted_story_id=predicted_story_id,
                        relationship=_relationship_label(assessment.relationship),
                        delta_type=_delta_label(assessment.change_type),
                        material=assessment.materiality >= 0.7,
                        display=display,
                        profile_relevance=relevance,
                        selected=selected,
                        reported_fact_ids=None,
                        latency_ms=round((perf_counter() - started) * 1000.0, 4),
                        metadata={
                            "adapter": self.name,
                            "identity_confidence": round(best_score, 4),
                            "delta_confidence": assessment.confidence,
                        },
                    )
                )
                stories[predicted_story_id] = _ObservedStory(
                    predicted_story_id=predicted_story_id,
                    identity=identity,
                    title=candidate.title,
                    last_seen=day.date,
                )
        return predictions


class LocalDeltaModelAdapter:
    """Evaluate an existing AI client on bounded source/baseline decisions.

    Retrieval and gold labels remain offline. The adapter deliberately evaluates
    delta decisions rather than asking the model to generate and then judge its
    own prose.
    """

    def __init__(
        self,
        client: AIClient,
        config: DeltaExtractionConfig,
        *,
        debug: DebugLogger | None = None,
        name: str = "local_delta_model",
        fixture_mode: str = "direct",
    ) -> None:
        self.client = client
        self.config = config
        self.debug = debug or DebugLogger(False)
        self.name = str(name or "local_delta_model")
        self.fixture_mode = _fixture_mode(fixture_mode)

    def predict(self, arc: EvalArcInput) -> List[EvalPrediction]:
        provider = FixtureNewsProvider(arc)
        extractor = DeltaExtractor(self.client, self.config, debug=self.debug)
        stories: Dict[str, _ObservedStory] = {}
        prior_reports: List[PriorReport] = []
        predictions: List[EvalPrediction] = []
        for day in arc.days:
            fetched = _fetch_fixture(provider, day.date, self.fixture_mode)
            selected: List[SelectedArticle] = []
            provisional_ids: Dict[str, str] = {}
            identities: Dict[str, StoryIdentity] = {}
            for candidate in fetched.candidates:
                identity = story_identity_for_candidate(candidate)
                best_score, best_story = _best_observed_story(identity, stories)
                provisional_id = (
                    best_story.predicted_story_id
                    if best_story is not None and best_score >= MATCH_CONFIDENCE_THRESHOLD
                    else _unused_story_id(identity.story_key, stories)
                )
                provisional_ids[candidate.id] = provisional_id
                identities[candidate.id] = identity
                set_memory_annotation(
                    candidate,
                    MemoryAnnotation(
                        story_key=provisional_id,
                        story_family_key=identity.story_family_key,
                        story_title=identity.story_title,
                        match_confidence=best_score,
                    ),
                )
                selected.append(
                    SelectedArticle(
                        candidate=candidate,
                        decision=HeadlineDecision(
                            candidate_id=candidate.id,
                            score=8.0,
                            topic="Personalized monitored stories",
                            personal_relevance=7.0,
                            impact=7.0,
                            novelty=7.0,
                            urgency=5.0,
                            confidence=6.0,
                        ),
                        article_text=provider.article_text(candidate.id),
                        extraction_status="fixture",
                    )
                )

            story_memory = _model_story_memory(selected, stories, day.date)
            started = perf_counter()
            model_error = ""
            try:
                packet = extractor.extract(
                    selected,
                    arc.profile,
                    [TopicConfig(name="Personalized monitored stories")],
                    prior_reports,
                    "Report profile-relevant material changes; suppress repeated facts.",
                    day.date,
                    story_memory=story_memory,
                    brief_name=arc.id,
                )
            except (AIJsonError, AITransportError, ValueError) as exc:
                model_error = str(exc)
                packet = {}
                extractor.warnings.append(f"model decision failed; deterministic fallback used: {exc}")
            call_latency_ms = (perf_counter() - started) * 1000.0
            decision_by_article = _model_decisions_by_article(packet)
            per_item_latency = call_latency_ms / max(1, len(selected))
            current_major: List[Dict[str, Any]] = []
            for article in selected:
                candidate = article.candidate
                raw = decision_by_article.get(candidate.id)
                if raw is None:
                    best_score, best_story = _best_observed_story(identities[candidate.id], stories)
                    fallback = assess_lexical_change(
                        candidate.title,
                        best_story.title if best_story is not None and best_score >= 0.34 else None,
                        story_key_match=best_story is not None and best_score >= MATCH_CONFIDENCE_THRESHOLD,
                    )
                    relationship = _relationship_label(fallback.relationship)
                    delta_type = _delta_label(fallback.change_type)
                    materiality = fallback.materiality
                    confidence = fallback.confidence
                    disposition = fallback.disposition
                    prior_story_key = best_story.predicted_story_id if relationship == "same_story" and best_story else ""
                    decision_summary = ""
                    fallback_used = True
                else:
                    relationship = _relationship_label(str(raw.get("relationship", "uncertain") or "uncertain"))
                    delta_type = _delta_label(str(raw.get("change_type", "uncertain") or "uncertain"))
                    materiality = _bounded(raw.get("materiality"), 0.0)
                    confidence = _bounded(raw.get("confidence"), 0.0)
                    disposition = str(raw.get("disposition", "uncertain") or "uncertain")
                    prior_story_key = str(raw.get("prior_story_key", "") or "").strip()
                    decision_summary = str(raw.get("summary", "") or "").strip()
                    fallback_used = False

                predicted_story_id = provisional_ids[candidate.id]
                if relationship == "same_story" and prior_story_key in stories:
                    predicted_story_id = prior_story_key
                display = _safe_model_display(
                    disposition=disposition,
                    relationship=relationship,
                    delta_type=delta_type,
                    confidence=confidence,
                )
                relevance = _profile_relevance(candidate, arc)
                predictions.append(
                    EvalPrediction(
                        arc_id=arc.id,
                        date=day.date,
                        document_id=candidate.id,
                        predicted_story_id=predicted_story_id,
                        relationship=relationship,
                        delta_type=delta_type,
                        material=materiality >= 0.7,
                        display=display,
                        profile_relevance=relevance,
                        selected=relevance != "irrelevant" and display != "omit",
                        reported_fact_ids=None,
                        latency_ms=round(per_item_latency, 4),
                        metadata={
                            "adapter": self.name,
                            "model_fallback_used": fallback_used,
                            "model_error": model_error,
                            "decision_summary": decision_summary,
                            "decision_confidence": round(confidence, 4),
                            "extractor_warnings": list(extractor.warnings),
                        },
                    )
                )
                stories[predicted_story_id] = _ObservedStory(
                    predicted_story_id=predicted_story_id,
                    identity=identities[candidate.id],
                    title=candidate.title,
                    last_seen=day.date,
                )
                current_major.append(
                    {
                        "headline": candidate.title,
                        "story_key": predicted_story_id,
                        "source": candidate.source,
                        "url": candidate.url,
                    }
                )
            prior_reports.insert(
                0,
                PriorReport(
                    id=f"fixture:{arc.id}:{day.date}",
                    date=day.date,
                    title=f"Fixture report {day.date}",
                    path="",
                    summary="",
                    major_headlines=current_major,
                ),
            )
            prior_reports = prior_reports[: max(1, int(self.config.max_prior_reports))]
        return predictions

    def diagnostics(self) -> Dict[str, Any]:
        return self.debug.analytics.payload()


class ScriptedOracleAdapter:
    """Harness self-test adapter. Never use its score as a model result."""

    name = "scripted_oracle"

    def __init__(self, corpus: EvalCorpus) -> None:
        self._expectations = corpus.expectations_by_key()

    def predict(self, arc: EvalArcInput) -> List[EvalPrediction]:
        output: List[EvalPrediction] = []
        for day in arc.days:
            for document in day.documents:
                expected = self._expectations[(arc.id, day.date, document.id)]
                output.append(
                    EvalPrediction(
                        arc_id=arc.id,
                        date=day.date,
                        document_id=document.id,
                        predicted_story_id=expected.canonical_story_id,
                        relationship=expected.relationship,
                        delta_type=expected.delta_type,
                        material=expected.material,
                        display=expected.display,
                        profile_relevance=expected.profile_relevance,
                        selected=expected.should_select,
                        reported_fact_ids=list(expected.required_fact_ids),
                    )
                )
        return output


class PredictionFileAdapter:
    name = "prediction_file"

    def __init__(self, predictions: List[EvalPrediction], *, name: str = "prediction_file") -> None:
        self.name = str(name or "prediction_file")
        self._by_arc: Dict[str, List[EvalPrediction]] = {}
        for prediction in predictions:
            self._by_arc.setdefault(prediction.arc_id, []).append(prediction)

    def predict(self, arc: EvalArcInput) -> List[EvalPrediction]:
        return list(self._by_arc.get(arc.id, []))


class FaultInjectionAdapter:
    """Deterministic negative controls for proving that metrics catch failures."""

    def __init__(self, base: EvaluationAdapter, mode: str) -> None:
        self.base = base
        self.mode = str(mode or "")
        self.name = f"fault:{self.mode}:{base.name}"

    def predict(self, arc: EvalArcInput) -> List[EvalPrediction]:
        rows = [
            replace(
                row,
                reported_fact_ids=(list(row.reported_fact_ids) if row.reported_fact_ids is not None else None),
                unsupported_claims=list(row.unsupported_claims),
                metadata=dict(row.metadata),
            )
            for row in self.base.predict(arc)
        ]
        if self.mode == "drop_every_other":
            return rows[::2]
        for row in rows:
            if self.mode == "merge_all_stories":
                row.predicted_story_id = f"{arc.id}:merged"
            elif self.mode == "omit_everything":
                row.display = "omit"
                row.selected = False
            elif self.mode == "unsupported_claims":
                row.unsupported_claims = ["Injected unsupported claim"]
            elif self.mode == "call_everything_new":
                row.delta_type = "new"
        if self.mode == "hallucinate_quiet_days":
            for day in arc.days:
                if day.documents:
                    continue
                rows.append(
                    EvalPrediction(
                        arc_id=arc.id,
                        date=day.date,
                        document_id="__invented_without_source__",
                        predicted_story_id=f"{arc.id}:invented:{day.date}",
                        relationship="new_story",
                        delta_type="new",
                        material=True,
                        display="full_report",
                        profile_relevance="must_select",
                        selected=True,
                        reported_fact_ids=[],
                        unsupported_claims=["Injected output on a source-empty day"],
                    )
                )
        return rows


def _best_observed_story(
    identity: StoryIdentity,
    stories: Dict[str, _ObservedStory],
) -> tuple[float, _ObservedStory | None]:
    best_score = 0.0
    best: _ObservedStory | None = None
    for story in stories.values():
        similarity = compare_token_sets(identity.tokens, story.identity.tokens)
        score = similarity.confidence
        if identity.story_key == story.identity.story_key and not similarity.numeric_conflict:
            score = 1.0
        if score > best_score:
            best_score = score
            best = story
    return best_score, best


def _unused_story_id(base: str, stories: Dict[str, _ObservedStory]) -> str:
    candidate = str(base or "story")
    if candidate not in stories:
        return candidate
    index = 2
    while f"{candidate}:{index}" in stories:
        index += 1
    return f"{candidate}:{index}"


def _profile_relevance(candidate, arc: EvalArcInput) -> str:
    signals = profile_match_signals(candidate, arc.profile)
    if signals.get("avoid_matches"):
        return "irrelevant"
    if signals.get("wants_matches") or signals.get("beat_matches") or signals.get("geo_matches"):
        return "must_select"
    return "eligible"


def _model_story_memory(
    selected: List[SelectedArticle],
    stories: Dict[str, _ObservedStory],
    date: str,
) -> Dict[str, Any]:
    baselines = [
        {
            "story_key": story.predicted_story_id,
            "story_family_key": story.identity.story_family_key,
            "title": story.title,
            "last_seen": story.last_seen,
            "last_change_type": "",
            "last_delta_summary": "",
            "knowns": [],
            "unknowns": [],
            "watch_signals": [],
        }
        for story in list(stories.values())[-8:]
    ]
    return {
        "schema_version": 1,
        "as_of_date": date,
        "stories": [
            {
                "story_key": provisional.story_key if provisional is not None else f"article:{article.candidate.id}",
                "current_title": article.candidate.title,
                "current_article_ids": [article.candidate.id],
                "current_articles": [{
                    "id": article.candidate.id,
                    "headline": article.candidate.title,
                    "source": article.candidate.source,
                    "excerpt": (article.article_text or article.candidate.snippet)[:600],
                }],
                "prior_baselines": baselines,
            }
            for article in selected
            for provisional in [article.candidate.annotations.memory]
        ],
    }


def _model_decisions_by_article(packet: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = packet.get("story_decisions", []) if isinstance(packet, dict) else []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("article_ids"), list):
            continue
        for article_id in row["article_ids"]:
            key = str(article_id or "").strip()
            if key:
                grouped.setdefault(key, []).append(row)
    output: Dict[str, Dict[str, Any]] = {}
    for article_id, decisions in grouped.items():
        signatures = {
            (
                str(item.get("relationship", "") or ""),
                str(item.get("change_type", "") or ""),
                str(item.get("disposition", "") or ""),
            )
            for item in decisions
        }
        if len(signatures) == 1:
            output[article_id] = max(decisions, key=lambda item: _bounded(item.get("confidence"), 0.0))
    return output


def _safe_model_display(
    *,
    disposition: str,
    relationship: str,
    delta_type: str,
    confidence: float,
) -> str:
    if disposition == "continuing_bullet":
        return "continuing_bullet"
    if (
        disposition == "omit"
        and relationship == "same_story"
        and delta_type == "unchanged"
        and confidence >= 0.7
    ):
        return "omit"
    return "full_report"


def _bounded(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _fixture_mode(value: str) -> str:
    normalized = str(value or "direct").strip().lower()
    if normalized not in {"direct", "rss"}:
        raise ValueError(f"unsupported fixture mode: {value!r}")
    return normalized


def _fetch_fixture(provider: FixtureNewsProvider, date: str, mode: str):
    return provider.fetch_via_rss(date) if mode == "rss" else provider.fetch(date)


def _relationship_label(value: str) -> str:
    return "new_story" if value == "distinct_story" else value


def _delta_label(value: str) -> str:
    if value in {"escalated", "weakened"}:
        return "material_update"
    return value
