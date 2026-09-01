from __future__ import annotations

"""Evaluate evidence-constrained story threading without corpus-specific rules.

The evaluator supports canonical-key routing over retained oracle history and
production-like chronological retrieval.  It runs structural-only and Qwen decision-only
variants, then optionally audits Qwen claim edges with AlignScore's published
three-way NLI head.  AlignScore uses argmax labels; no thresholds are fitted to
the evaluation corpus.
"""

import argparse
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mydailynews.ai.factory import create_ai_client  # noqa: E402
from mydailynews.analysis.claim_delta import request_from_claim_delta_payload  # noqa: E402
from mydailynews.analysis.deterministic_delta import (  # noqa: E402
    build_deterministic_delta_scaffold,
    merge_claim_delta_with_model,
)
from mydailynews.analysis.identity_gate import enforce_candidate_identity_gate  # noqa: E402
from mydailynews.app.config import load_config  # noqa: E402
from mydailynews.app.models import HeadlineDecision, MemoryAnnotation, MemoryConfig, SelectedArticle  # noqa: E402
from mydailynews.domain.candidate_annotations import (  # noqa: E402
    candidate_memory_annotation,
    set_memory_annotation,
)
from mydailynews.evaluation.schema import load_corpus  # noqa: E402
from mydailynews.memory.context import build_story_memory_context  # noqa: E402
from mydailynews.memory.ranking import annotate_candidates_with_memory  # noqa: E402
from mydailynews.memory.recall import apply_delta_signals_to_selected  # noqa: E402
from mydailynews.memory.story_retrieval import StoryCandidateMatch  # noqa: E402
from mydailynews.memory.story_store import StoryStore, story_baseline_payload  # noqa: E402


SEMANTIC_HASH_PATHS = (
    "mydailynews/ai/base.py",
    "mydailynews/ai/factory.py",
    "mydailynews/analysis/claim_delta.py",
    "mydailynews/analysis/deterministic_delta.py",
    "mydailynews/analysis/delta.py",
    "mydailynews/analysis/identity_gate.py",
    "mydailynews/ai/prompts.py",
    "mydailynews/ai/schemas.py",
    "mydailynews/ai/alignscore_nli.py",
    "mydailynews/ai/qwen_semantic_delta.py",
    "mydailynews/app/config.py",
    "mydailynews/app/models.py",
    "mydailynews/common/utils.py",
    "mydailynews/domain/candidate_annotations.py",
    "mydailynews/domain/text_similarity.py",
    "mydailynews/evaluation/schema.py",
    "mydailynews/memory/context.py",
    "mydailynews/memory/coverage.py",
    "mydailynews/memory/ranking.py",
    "mydailynews/memory/recall.py",
    "mydailynews/memory/story_retrieval.py",
    "mydailynews/memory/story_keys.py",
    "mydailynews/memory/story_store.py",
    "tools/run_semantic_thread_evaluation.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config.cpu-qwen1.7b-full.json",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=REPO_ROOT / "evals" / "cases" / "change_monitoring.v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("oracle", "full_stream"), default=("oracle", "full_stream"))
    parser.add_argument("--contexts", nargs="+", type=int, default=(2048, 4096))
    parser.add_argument("--max-arcs", type=int, default=0, help="Development smoke-test only; zero means all arcs.")
    parser.add_argument(
        "--qwen-model",
        type=Path,
        default=REPO_ROOT / "models" / "Qwen3-1.7B-Q4_K_M.gguf",
    )
    parser.add_argument(
        "--alignscore-checkpoint",
        type=Path,
        default=REPO_ROOT / "models" / "AlignScore-base" / "alignscore-base-nli.safetensors",
    )
    parser.add_argument(
        "--alignscore-tokenizer",
        type=Path,
        default=REPO_ROOT / "models" / "AlignScore-base" / "roberta-base",
    )
    parser.add_argument("--without-alignscore", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"output already exists (blind reports are immutable): {args.output}")
    if any(value < 1024 for value in args.contexts):
        parser.error("every context size must be at least 1024")

    corpus = load_corpus(args.corpus)
    if args.max_arcs > 0:
        corpus = replace(corpus, arcs=corpus.arcs[: args.max_arcs])
    app_config = load_config(args.config)
    server_runtime = server_model_snapshot(app_config.ai_summary.base_url)
    if args.qwen_model.name not in server_runtime["model_basenames"]:
        parser.error(
            "running llama.cpp server does not report the frozen Qwen model: "
            f"expected {args.qwen_model.name!r}, got {server_runtime['model_ids']!r}"
        )
    alignscore = None
    if not args.without_alignscore:
        from mydailynews.ai.alignscore_nli import AlignScoreNliScorer

        alignscore = AlignScoreNliScorer(
            args.alignscore_checkpoint,
            args.alignscore_tokenizer,
            batch_size=8,
        )

    all_rows: list[dict[str, Any]] = []
    all_audits: list[dict[str, Any]] = []
    state_sizes: list[dict[str, Any]] = []

    for mode in args.modes:
        rows, audits, sizes = replay(
            corpus,
            mode=mode,
            variant="structural_only",
            inferencer=None,
            alignscore=None,
        )
        all_rows.extend(rows)
        all_audits.extend(audits)
        state_sizes.extend(sizes)

    for context_tokens in dict.fromkeys(args.contexts):
        output_tokens = 128
        reserve_tokens = max(256, context_tokens // 16)
        input_tokens = context_tokens - output_tokens - reserve_tokens
        client_config = replace(
            app_config.ai_summary,
            context_window_tokens=context_tokens,
            max_input_tokens=input_tokens,
            max_new_tokens=output_tokens,
            temperature=0.0,
            top_p=1.0,
            json_retries=1,
        )
        client = create_ai_client(client_config)
        from mydailynews.ai.qwen_semantic_delta import QwenSemanticDeltaInferencer

        inferencer = QwenSemanticDeltaInferencer(
            client,
            max_current_claims=4 if context_tokens <= 2048 else 7,
            max_prior_claims=6 if context_tokens <= 2048 else 12,
            pair_scorer=alignscore,
            max_new_tokens=output_tokens,
            input_token_limit=input_tokens,
            label_prefix=f"semantic_delta.{context_tokens}",
        )
        try:
            for mode in args.modes:
                rows, audits, sizes = replay(
                    corpus,
                    mode=mode,
                    variant=f"qwen_evidence_{context_tokens}",
                    inferencer=inferencer,
                    alignscore=alignscore,
                )
                all_rows.extend(rows)
                all_audits.extend(audits)
                state_sizes.extend(sizes)
        finally:
            client.close()

    report = {
        "kind": "semantic_story_thread_evaluation.v1",
        "protocol": {
            "corpus_system_isolation": True,
            "no_semantic_regex_or_phrase_rules": True,
            "oracle_mode": (
                "canonical-key routing over normally retained history; semantic isolation only "
                "when the expected prior survives retention; not end-to-end retrieval"
            ),
            "oracle_retention_days": 30,
            "full_stream": "one global chronological store; each date is annotated before writeback",
            "evaluation_metadata_in_model_or_retrieval": False,
            "primary_metrics": "identity, relationship, delta, thread_delta, and pairwise clustering",
            "secondary_metrics": "materiality and display; fixture selection is bypassed",
            "semantic_cascade": (
                "one bounded candidate-identity call, then fixed-slot atomic claim-pair "
                "classification and deterministic ontology aggregation"
            ),
            "claim_pairing": (
                "non-headline current claims when body claims exist; each paired to the top two "
                "prior body claims by maximum bidirectional AlignScore entailment/contradiction "
                "probability; ranking only, with no fitted threshold"
            ),
            "materiality_policy": (
                "fixed policy baselines from relation aggregation, not separately evidence-scored; "
                "secondary diagnostic only"
            ),
            "materiality_cutoff": 0.7,
            "qwen_temperature": 0.0,
            "alignscore_policy": "three-way argmax; no fitted threshold",
            "nli_gate_history": "diagnostic only; shares the Qwen retrieval/writeback history",
            "modes": list(args.modes),
            "context_windows": list(dict.fromkeys(args.contexts)),
            "max_arcs": args.max_arcs,
        },
        "manifest": manifest(args),
        "server_runtime": server_runtime,
        "summary": summarize(all_rows),
        "nli_audit": summarize_audits(all_audits),
        "state_sizes": state_sizes,
        "rows": all_rows,
        "claim_relation_audits": all_audits,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(compact_console_summary(report["summary"]), indent=2))
    print(f"report={args.output / 'report.json'}")
    return 0


def replay(corpus, *, mode: str, variant: str, inferencer, alignscore):
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    sizes: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix=f"mydailynews-semantic-{mode}-{variant}-") as raw_root:
        root = Path(raw_root)
        store_path = root / "global-stream.json"
        store = StoryStore(store_path)
        canonical_to_predicted: dict[tuple[str, str], str] = {}
        for date_entries in chronological_batches(corpus):
            pending_writeback = []
            prepared = prepare_date_batch(
                date_entries,
                mode=mode,
                store=store,
            )
            for arc, day, document, expected, article, memory in prepared:
                scaffold = build_deterministic_delta_scaffold(
                    [article],
                    [],
                    story_memory=memory,
                    story_store=store,
                )
                model_packet: dict[str, Any] = {}
                model_raw: dict[str, Any] = {}
                model_requests = 0
                model_error = ""
                model_seconds = 0.0
                needs_model = any(
                    isinstance(item, dict)
                    and isinstance(item.get("claim_delta"), dict)
                    and bool(item["claim_delta"].get("requires_semantic_inference"))
                    for item in scaffold.get("story_decisions", [])
                )
                model_attempted = inferencer is not None and needs_model
                if model_attempted:
                    started = perf_counter()
                    try:
                        scaffold_row = scaffold["story_decisions"][0]
                        request = request_from_claim_delta_payload(
                            scaffold_row.get("claim_delta")
                        )
                        proposed = inferencer.compare(request) if request is not None else None
                        if proposed is not None:
                            model_packet = {
                                "story_decisions": [
                                    semantic_model_row(proposed, article_id=document.id)
                                ]
                            }
                    except Exception as exc:
                        model_error = f"{type(exc).__name__}: {exc}"
                    model_raw = dict(getattr(inferencer, "last_raw", {}) or {})
                    model_requests = int(getattr(inferencer, "last_request_count", 0) or 0)
                    model_seconds = perf_counter() - started
                model_no_decision = bool(model_attempted and not model_packet and not model_error)

                packet = merge_claim_delta_with_model(scaffold, model_packet)
                packet = enforce_candidate_identity_gate(packet, memory)
                apply_delta_signals_to_selected(selected=[article], delta_packet=packet)
                decision = packet["story_decisions"][0]
                annotation = candidate_memory_annotation(article.candidate)
                predicted_key = annotation.story_key if annotation is not None else ""
                canonical_key = (arc.id, expected.canonical_story_id)
                identity_correct = identity_score(
                    mode,
                    expected,
                    predicted_key,
                    canonical_key,
                    canonical_to_predicted,
                    relationship_name(decision.get("relationship")),
                )
                if canonical_key not in canonical_to_predicted:
                    canonical_to_predicted[canonical_key] = predicted_key

                audit = audit_model_decision(
                    model_packet,
                    scaffold,
                    article_id=document.id,
                    scorer=alignscore,
                )
                if audit:
                    audit.update(
                        {
                            "mode": mode,
                            "variant": variant,
                            "arc": arc.id,
                            "document_id": document.id,
                        }
                    )
                    audits.append(audit)

                base_row = result_row(
                    mode=mode,
                    variant=variant,
                    arc=arc,
                    document=document,
                    expected=expected,
                    decision=decision,
                    predicted_key=predicted_key,
                    identity_correct=identity_correct,
                    semantic_required=needs_model,
                    model_attempted=model_attempted,
                    model_no_decision=model_no_decision,
                    model_raw=model_raw,
                    model_requests=model_requests,
                    model_error=model_error,
                    model_seconds=model_seconds,
                    audit=audit,
                )
                rows.append(base_row)
                if inferencer is not None and alignscore is not None:
                    gated = nli_gated_decision(decision, audit)
                    gated_uncertain = gated.get("relationship") == "uncertain"
                    rows.append(
                        result_row(
                            mode=mode,
                            variant=f"{variant}_nli_gate",
                            arc=arc,
                            document=document,
                            expected=expected,
                            decision=gated,
                            predicted_key="" if gated_uncertain else predicted_key,
                            identity_correct=(
                                identity_correct
                                if not gated_uncertain
                                else expected.relationship == "uncertain"
                            ),
                            semantic_required=needs_model,
                            model_attempted=model_attempted,
                            model_no_decision=model_no_decision,
                            model_raw=model_raw,
                            model_requests=model_requests,
                            model_error=model_error,
                            model_seconds=model_seconds,
                            audit=audit,
                        )
                    )

                pending_writeback.append((article, day.date, packet))

            # Production retrieves/annotates a day's batch against the prior
            # day's frozen state.  Same-day documents cannot become artificial
            # temporal predecessors merely because of fixture ordering.
            for article, date, packet in pending_writeback:
                store.update_selected(
                    selected=[article],
                    date=date,
                    delta_packet=packet,
                )
        sizes.append(
            {
                "mode": mode,
                "variant": variant,
                "arc": "__global_timeline__",
                "bytes": store_path.stat().st_size if store_path.exists() else 0,
                "stories": len(store.records()),
            }
        )
    return rows, audits, sizes


def chronological_documents(corpus):
    """Interleave every event family without exposing its partition to retrieval."""

    entries = []
    for arc in corpus.arcs:
        for day in arc.days:
            expected_by_id = {item.document_id: item for item in day.expectations}
            for document in day.documents:
                entries.append(
                    (day.date, document.published_at, document.id, arc, day, document, expected_by_id[document.id])
                )
    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(arc, day, document, expected) for _, _, _, arc, day, document, expected in entries]


def chronological_batches(corpus):
    batches = []
    for entry in chronological_documents(corpus):
        date = entry[1].date
        if not batches or batches[-1][0][1].date != date:
            batches.append([])
        batches[-1].append(entry)
    return batches


def prepare_date_batch(entries, *, mode, store):
    if mode == "oracle":
        prepared = []
        for arc, day, document, expected in entries:
            article, memory = prepare_oracle_article(
                arc_id=arc.id,
                document=document,
                expected=expected,
                date=day.date,
                store=store,
            )
            prepared.append((arc, day, document, expected, article, memory))
        return prepared

    candidates = [production_candidate(entry[2]) for entry in entries]
    decisions = {
        candidate.id: HeadlineDecision(candidate.id, score=8.0)
        for candidate in candidates
    }
    date = entries[0][1].date
    annotate_candidates_with_memory(
        candidates=candidates,
        decisions=decisions,
        memory_config=MemoryConfig(),
        coverage_store=None,
        story_store=store,
        date=date,
    )
    prepared = []
    for (arc, day, document, expected), candidate in zip(entries, candidates):
        article = SelectedArticle(
            candidate=candidate,
            decision=decisions[candidate.id],
            article_text=document.body,
            extraction_status="evaluation_fixture",
        )
        memory = build_story_memory_context(
            selected=[article],
            story_groups=[],
            story_store=store,
            coverage_store=None,
            prior_reports=[],
            date=day.date,
        )
        prepared.append((arc, day, document, expected, article, memory))
    return prepared


def semantic_model_row(decision, *, article_id):
    return {
        "article_ids": [article_id],
        "prior_story_key": decision.prior_story_key,
        "relationship": decision.relationship,
        "change_type": decision.change_type,
        "materiality": decision.materiality,
        "confidence": decision.confidence,
        "disposition": decision.disposition,
        "summary": decision.summary,
        "current_evidence_ids": list(decision.current_evidence_ids),
        "prior_evidence_ids": list(decision.prior_evidence_ids),
        "superseded_prior_evidence_ids": list(
            decision.superseded_prior_evidence_ids
        ),
        "claim_relations": [item.payload() for item in decision.claim_relations],
    }


def prepare_oracle_article(*, arc_id, document, expected, date, store):
    candidate = production_candidate(document)
    headline = HeadlineDecision(candidate.id, score=8.0)
    story_key = f"oracle:{arc_id}:{expected.canonical_story_id}"
    prior = next((item for item in store.records() if item.story_key == story_key), None)
    set_memory_annotation(
        candidate,
        MemoryAnnotation(
            story_key=story_key,
            story_family_key=f"oracle:{arc_id}",
            story_title=candidate.title,
            match_confidence=1.0,
        ),
    )
    baselines = []
    if prior is not None:
        match = StoryCandidateMatch(
            score=1.0,
            record=prior,
            lexical_score=1.0,
            alias_score=1.0,
            entity_score=1.0,
            event_score=1.0,
            fact_score=1.0,
            numeric_conflict=False,
        )
        baselines = [story_baseline_payload(match, max_facts=4)]
    memory = {
        "schema_version": 2,
        "as_of_date": date,
        "stories": [
            {
                "story_key": story_key,
                "current_title": candidate.title,
                "current_article_ids": [candidate.id],
                "prior_baselines": baselines,
            }
        ],
    }

    article = SelectedArticle(
        candidate=candidate,
        decision=headline,
        article_text=document.body,
        extraction_status="evaluation_fixture",
    )
    return article, memory


def production_candidate(document):
    """Return only fields that a fetched article supplies in production.

    Evaluation categories and tags describe corpus strata and adversarial
    properties.  They are useful for scoring slices but must never become
    retrieval evidence.
    """

    return replace(
        document.to_candidate(),
        category="",
        tags=[],
        metadata={},
    )


def identity_score(
    mode,
    expected,
    predicted_key,
    canonical_key,
    canonical_to_predicted,
    predicted_relationship,
):
    if mode == "oracle":
        return predicted_relationship == expected.relationship
    if expected.relationship == "same_story":
        return bool(
            canonical_to_predicted.get(canonical_key)
            and predicted_key == canonical_to_predicted[canonical_key]
        )
    if expected.relationship == "new_story":
        return bool(predicted_key and predicted_key not in set(canonical_to_predicted.values()))
    return predicted_relationship == expected.relationship


def result_row(
    *,
    mode,
    variant,
    arc,
    document,
    expected,
    decision,
    predicted_key,
    identity_correct,
    semantic_required,
    model_attempted,
    model_no_decision,
    model_raw,
    model_requests,
    model_error,
    model_seconds,
    audit,
):
    predicted_material = float(decision.get("materiality", 0.0) or 0.0) >= 0.7
    predicted_relationship = relationship_name(decision.get("relationship"))
    relationship_correct = predicted_relationship == expected.relationship
    delta_correct = str(decision.get("change_type", "")) == expected.delta_type
    material_correct = predicted_material == expected.material
    display_correct = str(decision.get("disposition", "")) == expected.display
    semantic_validation = (
        decision.get("semantic_validation", {})
        if isinstance(decision.get("semantic_validation"), dict)
        else {}
    )
    return {
        "mode": mode,
        "variant": variant,
        "arc": arc.id,
        "split": arc.split,
        "arc_tags": list(arc.tags),
        "category": document.category,
        "document_tags": list(document.tags),
        "document_id": document.id,
        "canonical_cluster_id": f"{arc.id}:{expected.canonical_story_id}",
        "predicted_cluster_id": predicted_key,
        "is_continuation": expected.relationship == "same_story",
        "should_select": expected.should_select,
        "expected_relationship": expected.relationship,
        "predicted_relationship": predicted_relationship,
        "identity_correct": bool(identity_correct),
        "relationship_correct": relationship_correct,
        "expected_delta": expected.delta_type,
        "predicted_delta": str(decision.get("change_type", "")),
        "delta_correct": delta_correct,
        "expected_material": expected.material,
        "predicted_material": predicted_material,
        "material_correct": material_correct,
        "expected_display": expected.display,
        "predicted_display": str(decision.get("disposition", "")),
        "display_correct": display_correct,
        "thread_delta_correct": bool(identity_correct and relationship_correct and delta_correct),
        "joint_correct": bool(
            identity_correct
            and relationship_correct
            and delta_correct
            and material_correct
            and display_correct
        ),
        "abstained": str(decision.get("relationship", "")) == "uncertain" or str(decision.get("change_type", "")) == "uncertain",
        "decision_basis": str(decision.get("claim_delta", {}).get("decision_basis", "")) if isinstance(decision.get("claim_delta"), dict) else "",
        "semantic_required": bool(semantic_required),
        "model_attempted": bool(model_attempted),
        "model_no_decision": bool(model_no_decision),
        "model_raw": dict(model_raw) if isinstance(model_raw, dict) else {},
        "model_requests": int(model_requests),
        "model_error": model_error,
        "model_seconds": round(model_seconds, 4),
        "semantic_validation_status": str(semantic_validation.get("status", "") or ""),
        "semantic_validation_errors": list(semantic_validation.get("errors", []))
        if isinstance(semantic_validation.get("errors"), list)
        else [],
        "nli_all_checkable_edges_agree": audit.get("all_checkable_edges_agree") if audit else None,
    }


def audit_model_decision(model_packet, scaffold, *, article_id, scorer):
    if scorer is None or not isinstance(model_packet, dict):
        return {}
    model_row = next(
        (
            item
            for item in model_packet.get("story_decisions", [])
            if isinstance(item, dict) and article_id in item.get("article_ids", [])
        ),
        None,
    )
    scaffold_row = next(
        (
            item
            for item in scaffold.get("story_decisions", [])
            if isinstance(item, dict) and article_id in item.get("article_ids", [])
        ),
        None,
    )
    if model_row is None or scaffold_row is None:
        return {}
    model_labels = {
        "model_relationship": str(model_row.get("relationship", "") or ""),
        "model_change_type": str(model_row.get("change_type", "") or ""),
        "model_relation_types": [
            str(item.get("relation", "") or "")
            for item in model_row.get("claim_relations", [])
            if isinstance(item, dict)
        ],
    }
    request = request_from_claim_delta_payload(scaffold_row.get("claim_delta"))
    if request is None:
        return model_labels
    current = {item.claim_id: item.text for item in request.current_claims}
    prior = {item.claim_id: item.text for item in request.prior_claims}
    work: list[tuple[dict[str, Any], str, str]] = []
    for relation in model_row.get("claim_relations", []):
        if not isinstance(relation, dict):
            continue
        current_id = str(relation.get("current_claim_id", "") or "")
        prior_id = str(relation.get("prior_claim_id", "") or "")
        if current_id in current and prior_id in prior:
            work.append((relation, current[current_id], prior[prior_id]))
    if not work:
        return {
            **model_labels,
            "checkable_edges": 0,
            "agreements": 0,
            "all_checkable_edges_agree": None,
            "edges": [],
        }
    try:
        scores = scorer.score_bidirectional(
            [item[1] for item in work],
            [item[2] for item in work],
        )
    except Exception as exc:
        return {
            **model_labels,
            "checkable_edges": 0,
            "agreements": 0,
            "all_checkable_edges_agree": False,
            "error": f"{type(exc).__name__}: {exc}",
            "edges": [],
        }
    edges = []
    for (relation, _, _), score in zip(work, scores):
        relation_name = str(relation.get("relation", "") or "")
        forward = score.current_to_prior.predicted_label
        reverse = score.prior_to_current.predicted_label
        agreement = relation_agrees(relation_name, forward, reverse)
        edges.append(
            {
                "relation": relation_name,
                "forward_label": forward,
                "reverse_label": reverse,
                "agreement": agreement,
                "forward": score.current_to_prior.__dict__,
                "reverse": score.prior_to_current.__dict__,
            }
        )
    checkable = [item for item in edges if item["agreement"] is not None]
    return {
        **model_labels,
        "checkable_edges": len(checkable),
        "agreements": sum(item["agreement"] is True for item in checkable),
        "all_checkable_edges_agree": (
            all(item["agreement"] is True for item in checkable) if checkable else None
        ),
        "edges": edges,
    }


def relation_agrees(relation, forward, reverse):
    """Formal relation/NLI consistency; no vocabulary or corpus signals."""

    if relation == "equivalent":
        return forward == "entailment" and reverse == "entailment"
    if relation == "supports":
        return forward == "entailment"
    if relation == "adds_detail":
        return forward == "entailment" and reverse != "entailment"
    if relation == "weaker_restatement":
        return reverse == "entailment" and forward != "entailment"
    if relation in {"contradicts", "supersedes"}:
        return "contradiction" in {forward, reverse}
    return None


def nli_gated_decision(decision, audit):
    output = deepcopy(decision)
    if audit and audit.get("all_checkable_edges_agree") is False:
        output.update(
            {
                "prior_story_key": "",
                "relationship": "uncertain",
                "change_type": "uncertain",
                "materiality": 0.5,
                "confidence": 0.0,
                "disposition": "full_report",
                "summary": "The independent NLI edge audit disagreed; retain visibly as uncertain.",
            }
        )
    return output


def relationship_name(value):
    text = str(value or "")
    return "new_story" if text == "distinct_story" else text


def summarize(rows):
    output = {}
    keys = sorted({(row["mode"], row["variant"]) for row in rows})
    for mode, variant in keys:
        group = [row for row in rows if row["mode"] == mode and row["variant"] == variant]
        output[f"{mode}:{variant}"] = metrics(group)
        categories = sorted({str(row["category"] or "uncategorized") for row in group})
        output[f"{mode}:{variant}"]["by_category"] = {
            category: metrics([row for row in group if str(row["category"] or "uncategorized") == category])
            for category in categories
        }
        tags = sorted({tag for row in group for tag in [*row["arc_tags"], *row["document_tags"]]})
        output[f"{mode}:{variant}"]["by_tag"] = {
            tag: metrics(
                [
                    row
                    for row in group
                    if tag in row["arc_tags"] or tag in row["document_tags"]
                ]
            )
            for tag in tags
        }
    return output


def metrics(rows):
    calls = [row for row in rows if row["model_attempted"]]
    semantic_rows = [row for row in rows if row["semantic_required"]]
    accepted = [
        row
        for row in calls
        if row["semantic_validation_status"] == "accepted"
    ]
    continuations = [row for row in rows if row["is_continuation"]]
    selected = [row for row in rows if row["should_select"]]
    return {
        "documents": len(rows),
        "continuations": len(continuations),
        "identity_accuracy": ratio(row["identity_correct"] for row in rows),
        "relationship_accuracy": ratio(row["relationship_correct"] for row in rows),
        "delta_accuracy": ratio(row["delta_correct"] for row in rows),
        "thread_delta_accuracy": ratio(row["thread_delta_correct"] for row in rows),
        "materiality_accuracy": ratio(row["material_correct"] for row in rows),
        "display_accuracy": ratio(row["display_correct"] for row in rows),
        "display_scored_documents": len(selected),
        "display_accuracy_selected": ratio(row["display_correct"] for row in selected),
        "joint_accuracy": ratio(row["joint_correct"] for row in rows),
        "abstention_rate": ratio(row["abstained"] for row in rows),
        "semantic_required_documents": len(semantic_rows),
        "semantic_required_thread_delta_accuracy": ratio(
            row["thread_delta_correct"] for row in semantic_rows
        ),
        "continuation_delta_accuracy": ratio(row["delta_correct"] for row in continuations),
        "model_calls": len(calls),
        "model_requests": sum(row["model_requests"] for row in calls),
        "model_errors": sum(bool(row["model_error"]) for row in calls),
        "model_no_decisions": sum(row["model_no_decision"] for row in calls),
        "accepted_decisions": len(accepted),
        "accepted_coverage": round(len(accepted) / max(1, len(calls)), 4),
        "accepted_thread_delta_accuracy": ratio(
            row["thread_delta_correct"] for row in accepted
        ),
        "model_seconds_total": round(sum(row["model_seconds"] for row in calls), 3),
        "model_seconds_mean": round(sum(row["model_seconds"] for row in calls) / max(1, len(calls)), 3),
        "relationship_confusion": confusion(
            rows,
            expected="expected_relationship",
            predicted="predicted_relationship",
        ),
        "continuation_delta_confusion": confusion(
            continuations,
            expected="expected_delta",
            predicted="predicted_delta",
        ),
        "pairwise_clustering": pairwise_clustering(rows),
    }


def confusion(rows, *, expected, predicted):
    output = {}
    for row in rows:
        expected_value = str(row[expected] or "<empty>")
        predicted_value = str(row[predicted] or "<empty>")
        output.setdefault(expected_value, {})
        output[expected_value][predicted_value] = (
            output[expected_value].get(predicted_value, 0) + 1
        )
    return output


def pairwise_clustering(rows):
    true_positive = false_positive = false_negative = true_negative = 0
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            expected_same = left["canonical_cluster_id"] == right["canonical_cluster_id"]
            predicted_same = bool(
                left["predicted_cluster_id"]
                and left["predicted_cluster_id"] == right["predicted_cluster_id"]
            )
            if expected_same and predicted_same:
                true_positive += 1
            elif expected_same:
                false_negative += 1
            elif predicted_same:
                false_positive += 1
            else:
                true_negative += 1
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "true_positive": true_positive,
        "false_merge": false_positive,
        "false_split": false_negative,
        "true_negative": true_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(1e-12, precision + recall), 4),
    }


def summarize_audits(audits):
    output = {}
    for key in sorted({(item["mode"], item["variant"]) for item in audits}):
        group = [item for item in audits if (item["mode"], item["variant"]) == key]
        edges = sum(int(item.get("checkable_edges", 0)) for item in group)
        agreements = sum(int(item.get("agreements", 0)) for item in group)
        decisions = [item for item in group if item.get("all_checkable_edges_agree") is not None]
        output[f"{key[0]}:{key[1]}"] = {
            "decisions": len(group),
            "decisions_with_checkable_edges": len(decisions),
            "decisions_all_edges_agree": sum(item.get("all_checkable_edges_agree") is True for item in decisions),
            "checkable_edges": edges,
            "agreeing_edges": agreements,
            "edge_agreement": round(agreements / max(1, edges), 4),
        }
    return output


def ratio(values):
    values = list(values)
    return round(sum(bool(value) for value in values) / max(1, len(values)), 4)


def compact_console_summary(summary):
    """Keep interactive output useful without repeating all report slices."""

    fields = (
        "documents",
        "identity_accuracy",
        "relationship_accuracy",
        "delta_accuracy",
        "thread_delta_accuracy",
        "semantic_required_documents",
        "semantic_required_thread_delta_accuracy",
        "continuation_delta_accuracy",
        "abstention_rate",
        "accepted_coverage",
        "model_calls",
        "model_requests",
        "model_errors",
    )
    output = {}
    for name, row in summary.items():
        compact = {field: row.get(field) for field in fields}
        clustering = row.get("pairwise_clustering", {})
        compact["pairwise_clustering"] = {
            field: clustering.get(field)
            for field in ("precision", "recall", "f1", "false_merge", "false_split")
        }
        output[name] = compact
    return output


def manifest(args):
    paths = {name: REPO_ROOT / name for name in SEMANTIC_HASH_PATHS}
    paths.update(
        {
            "corpus": args.corpus,
            "config": args.config,
            "qwen_model": args.qwen_model,
        }
    )
    if not args.without_alignscore:
        paths["alignscore_checkpoint"] = args.alignscore_checkpoint
        for tokenizer_name in (
            "config.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ):
            paths[f"alignscore_tokenizer/{tokenizer_name}"] = (
                args.alignscore_tokenizer / tokenizer_name
            )
    return {
        name: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for name, path in paths.items()
        if path.is_file()
    }


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def server_model_snapshot(base_url):
    endpoint = f"{str(base_url).rstrip('/')}/models"
    with urlopen(endpoint, timeout=10) as response:  # noqa: S310 - localhost model server
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    model_ids = [
        str(item.get("id", "") or "")
        for item in rows
        if isinstance(item, dict)
    ]
    return {
        "endpoint": endpoint,
        "model_ids": model_ids,
        "model_basenames": [Path(item.replace("\\", "/")).name for item in model_ids],
        "metadata": [item.get("meta", {}) for item in rows if isinstance(item, dict)],
        "binding": "server-reported model basename plus SHA-256 of the intended local GGUF",
    }


def render_markdown(report):
    lines = [
        "# Evidence-constrained semantic story-thread evaluation",
        "",
        "Primary diagnostic metrics are story identity, relationship, delta, their combined thread+delta score, and pairwise clustering. Materiality/display are secondary because this harness bypasses upstream selection. Oracle mode routes by canonical key but still applies normal 30-day retention, so it isolates semantics only when a prior survives and is not an end-to-end retrieval result.",
        "",
        "| Mode / variant | Docs | Identity | Relation | Delta | Thread+delta | Semantic subset | Pair F1 | Accepted coverage | Abstain | Comparisons | Requests | No decision | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in report["summary"].items():
        lines.append(
            f"| {name} | {item['documents']} | {item['identity_accuracy']:.4f} | "
            f"{item['relationship_accuracy']:.4f} | {item['delta_accuracy']:.4f} | "
            f"{item['thread_delta_accuracy']:.4f} | "
            f"{item['semantic_required_thread_delta_accuracy']:.4f} | "
            f"{item['pairwise_clustering']['f1']:.4f} | {item['accepted_coverage']:.4f} | "
            f"{item['abstention_rate']:.4f} | {item['model_calls']} | {item['model_requests']} | "
            f"{item['model_no_decisions']} | {item['model_errors']} |"
        )
    lines.extend(["", "## AlignScore relation audit", ""])
    if report["nli_audit"]:
        lines.extend(
            [
                "| Mode / variant | Decisions | Checkable decisions | All agree | Edges | Edge agreement |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, item in report["nli_audit"].items():
            lines.append(
                f"| {name} | {item['decisions']} | {item['decisions_with_checkable_edges']} | "
                f"{item['decisions_all_edges_agree']} | {item['checkable_edges']} | "
                f"{item['edge_agreement']:.4f} |"
            )
    else:
        lines.append("AlignScore was disabled.")
    lines.extend(
        [
            "",
            "The `_nli_gate` rows are post-hoc safety diagnostics sharing the corresponding Qwen history; they are not independent chronological replays.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
