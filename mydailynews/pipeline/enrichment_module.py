from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from mydailynews.app.models import BriefOutput, EnrichmentOutput, SelectedArticle
from mydailynews.briefing.output import write_json
from mydailynews.common.utils import normalize_url
from mydailynews.common.warnings import extend_warnings
from mydailynews.enrichment.runner import StoryThreadEnricher
from mydailynews.pipeline.article_pipeline import populate_article_texts, record_enrichment_metrics
from mydailynews.pipeline.handoff import (
    STRUCTURED_BRIEF_NAMES,
    handoff_path,
    load_brief_handoff,
    load_brief_json,
    selected_article_to_handoff_payload,
    selected_articles_from_brief_json,
)
from mydailynews.pipeline.stage_artifacts import to_jsonable
from mydailynews.retrieval.article import ArticleRetriever


ENRICHMENT_OUTPUT_SCHEMA_VERSION = "enrichment_output.v1"


@dataclass
class EnrichmentInputSet:
    selected_articles: List[SelectedArticle]
    source_briefs: List[str]
    input_mode: Dict[str, str]
    warnings: List[str]


def run_enrichment(
    orchestrator,
    *,
    date: str,
    source_outputs: List[BriefOutput] | None = None,
    allow_disk_fallback: bool = True,
) -> EnrichmentOutput | None:
    if not _enrichment_enabled(orchestrator.config):
        warning = "enrichment: module is disabled by config; skipped."
        orchestrator.warnings.append(warning)
        orchestrator.debug.set_metric("module.enrichment.status", "skipped_disabled")
        return None

    run_warnings: List[str] = []
    output_dir = Path(orchestrator.config.output_dir)
    inputs = collect_enrichment_inputs(
        output_dir=output_dir,
        date=date,
        article_text_cache=getattr(orchestrator, "article_text_cache", None),
        source_outputs=list(source_outputs or []),
        allow_disk_fallback=allow_disk_fallback,
    )
    extend_warnings(run_warnings, inputs.warnings)
    if not inputs.selected_articles:
        warning = f"enrichment: no same-day general or detailed brief inputs were available for {date}."
        run_warnings.append(warning)
        extend_warnings(orchestrator.warnings, run_warnings)
        orchestrator.debug.set_metric("module.enrichment.status", "skipped_no_inputs")
        return None

    _refetch_degraded_article_texts(orchestrator, inputs.selected_articles, run_warnings)

    orchestrator.reporter.phase("Running standalone story enrichment...")
    orchestrator.debug.set_metric("module.enrichment.status", "running")
    orchestrator.debug.log(
        "enrichment.module",
        "starting",
        date=date,
        selected=len(inputs.selected_articles),
        source_briefs=",".join(inputs.source_briefs),
    )

    enricher = StoryThreadEnricher(
        orchestrator.config,
        http_cache=getattr(orchestrator, "enrichment_cache", None),
        debug=orchestrator.debug,
        ai_client=getattr(orchestrator, "summary_ai_client", None),
        cache=getattr(orchestrator, "synth_cache", None),
        brief_name="module",
        date=date,
    )
    with orchestrator.debug.span("module.enrichment"):
        enricher.enrich_many(inputs.selected_articles)
    extend_warnings(run_warnings, enricher.warnings)
    record_enrichment_metrics(
        brief_name="module",
        selected=inputs.selected_articles,
        debug=orchestrator.debug,
        story_thread_counts=(
            int(getattr(enricher, "story_threads_created", 0)),
            int(getattr(enricher, "story_threads_enriched", 0)),
            int(getattr(enricher, "story_threads_skipped", 0)),
        ),
    )

    payload = build_enrichment_payload(
        date=date,
        inputs=inputs,
        selected_articles=inputs.selected_articles,
        enricher_artifact=dict(getattr(enricher, "artifact", {}) or {}),
        warnings=run_warnings,
    )
    markdown_path, json_path = write_enrichment_outputs(output_dir, date, payload)
    _record_enrichment_artifact(
        orchestrator,
        payload=payload,
        markdown_path=markdown_path,
        json_path=json_path,
    )

    story_threads = payload.get("story_threads", [])
    orchestrator.debug.set_metric("module.enrichment.status", "completed")
    orchestrator.debug.log(
        "enrichment.module",
        "complete",
        markdown=markdown_path,
        json=json_path,
        selected=len(inputs.selected_articles),
        story_threads=len(story_threads) if isinstance(story_threads, list) else 0,
        warnings=len(run_warnings),
    )
    extend_warnings(orchestrator.warnings, run_warnings)
    return EnrichmentOutput(
        name="enrichment",
        json_path=str(json_path),
        markdown_path=str(markdown_path),
        source_briefs=inputs.source_briefs,
        selected_count=len(inputs.selected_articles),
        story_thread_count=len(story_threads) if isinstance(story_threads, list) else 0,
        warnings=run_warnings,
    )


def collect_enrichment_inputs(
    *,
    output_dir: Path,
    date: str,
    article_text_cache: Any | None = None,
    source_outputs: List[BriefOutput] | None = None,
    allow_disk_fallback: bool = True,
) -> EnrichmentInputSet:
    warnings: List[str] = []
    input_mode: Dict[str, str] = {}
    collected: List[SelectedArticle] = []
    outputs_by_name: Dict[str, BriefOutput] = {}
    for output in source_outputs or []:
        name = str(output.name or "").strip().lower()
        if name in STRUCTURED_BRIEF_NAMES:
            outputs_by_name[name] = output

    for brief_name in STRUCTURED_BRIEF_NAMES:
        selected: List[SelectedArticle] = []
        mode = "missing"
        source_output = outputs_by_name.get(brief_name)
        if source_output is not None:
            selected, mode = _selected_from_brief_output(
                source_output,
                brief_name=brief_name,
                article_text_cache=article_text_cache,
                warnings=warnings,
            )
        elif allow_disk_fallback:
            selected, mode = _selected_from_disk(
                output_dir=output_dir,
                date=date,
                brief_name=brief_name,
                article_text_cache=article_text_cache,
                warnings=warnings,
            )
        for article in selected:
            _record_article_source(article, brief_name, mode)
        input_mode[brief_name] = mode
        collected.extend(selected)

    deduped = _dedupe_articles(collected)
    source_briefs = []
    for name in STRUCTURED_BRIEF_NAMES:
        if any(name in _article_source_briefs(article) for article in deduped):
            source_briefs.append(name)
    return EnrichmentInputSet(
        selected_articles=deduped,
        source_briefs=source_briefs,
        input_mode=input_mode,
        warnings=warnings,
    )


def _selected_from_brief_output(
    output: BriefOutput,
    *,
    brief_name: str,
    article_text_cache: Any | None,
    warnings: List[str],
) -> tuple[List[SelectedArticle], str]:
    mode = "missing"
    handoff = Path(str(output.handoff_path or "").strip()) if str(output.handoff_path or "").strip() else None
    if handoff is not None and handoff.exists():
        try:
            result = load_brief_handoff(handoff)
            extend_warnings(warnings, result.warnings)
            mode = "handoff"
            if result.selected_articles:
                return result.selected_articles, mode
        except Exception as exc:
            warnings.append(f"enrichment: failed to load current-run {brief_name} handoff ({type(exc).__name__}: {exc}).")

    json_path = Path(str(output.json_path or "").strip()) if str(output.json_path or "").strip() else None
    if json_path is not None and json_path.exists():
        return _selected_from_brief_json(
            json_path,
            brief_name=brief_name,
            article_text_cache=article_text_cache,
            warnings=warnings,
            current_run=True,
        )
    return [], mode


def _selected_from_disk(
    *,
    output_dir: Path,
    date: str,
    brief_name: str,
    article_text_cache: Any | None,
    warnings: List[str],
) -> tuple[List[SelectedArticle], str]:
    mode = "missing"
    handoff = handoff_path(output_dir, date, brief_name)
    if handoff.exists():
        try:
            result = load_brief_handoff(handoff)
            extend_warnings(warnings, result.warnings)
            mode = "handoff"
            if result.selected_articles:
                return result.selected_articles, mode
        except Exception as exc:
            warnings.append(f"enrichment: failed to load {brief_name} handoff ({type(exc).__name__}: {exc}).")

    brief_json = output_dir / f"{date}_{brief_name}_brief.json"
    if brief_json.exists():
        return _selected_from_brief_json(
            brief_json,
            brief_name=brief_name,
            article_text_cache=article_text_cache,
            warnings=warnings,
            current_run=False,
        )
    return [], mode


def _selected_from_brief_json(
    json_path: Path,
    *,
    brief_name: str,
    article_text_cache: Any | None,
    warnings: List[str],
    current_run: bool,
) -> tuple[List[SelectedArticle], str]:
    try:
        payload = load_brief_json(json_path)
        return (
            selected_articles_from_brief_json(
                payload,
                brief_name=brief_name,
                json_path=json_path,
                article_text_cache=article_text_cache,
            ),
            "rehydrated_brief",
        )
    except Exception as exc:
        scope = "current-run " if current_run else ""
        warnings.append(f"enrichment: failed to rehydrate {scope}{brief_name} brief ({type(exc).__name__}: {exc}).")
        return [], "missing"


def build_enrichment_payload(
    *,
    date: str,
    inputs: EnrichmentInputSet,
    selected_articles: List[SelectedArticle],
    enricher_artifact: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    story_threads = enricher_artifact.get("story_threads", [])
    if not isinstance(story_threads, list):
        story_threads = []
    return {
        "schema_version": ENRICHMENT_OUTPUT_SCHEMA_VERSION,
        "date": date,
        "source_briefs": inputs.source_briefs,
        "input_mode": inputs.input_mode,
        "selected_articles": [
            selected_article_to_handoff_payload(article)
            for article in selected_articles
        ],
        "story_threads": story_threads,
        "context_sources_by_article": {
            article.candidate.id: [to_jsonable(asdict(source)) for source in article.context_sources]
            for article in selected_articles
        },
        "warnings": list(warnings),
        "metadata": {
            "selected_count": len(selected_articles),
            "context_source_count": sum(len(article.context_sources) for article in selected_articles),
            "story_threads_created": int(enricher_artifact.get("story_threads_created", 0) or 0),
            "enricher": enricher_artifact,
        },
    }


def write_enrichment_outputs(output_dir: Path, date: str, payload: Dict[str, Any]) -> tuple[Path, Path]:
    markdown_path = output_dir / f"{date}_enrichment.md"
    json_path = output_dir / f"{date}_enrichment.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_enrichment_markdown(payload), encoding="utf-8")
    write_json(json_path, payload)
    return markdown_path, json_path


def render_enrichment_markdown(payload: Dict[str, Any]) -> str:
    date = str(payload.get("date", "") or "")
    lines = [f"# Story Enrichment - {date}", ""]
    source_briefs = payload.get("source_briefs", [])
    if source_briefs:
        lines.append(f"_Source briefs: {', '.join(str(item) for item in source_briefs)}_")
        lines.append("")
    story_threads = payload.get("story_threads", [])
    if isinstance(story_threads, list) and story_threads:
        lines.append("## Story Threads")
        for thread in story_threads:
            if not isinstance(thread, dict):
                continue
            title = str(thread.get("story_title", "") or thread.get("story_id", "") or "Story")
            status = str(thread.get("status", "") or "")
            lines.append(f"### {title}")
            if status:
                lines.append(f"Status: {status}")
            internal_articles = thread.get("internal_articles", [])
            if isinstance(internal_articles, list):
                for item in internal_articles:
                    if not isinstance(item, dict):
                        continue
                    item_title = str(item.get("title", "") or "").strip()
                    summary = str(item.get("summary", "") or "").strip()
                    if item_title:
                        lines.append(f"- {item_title}")
                    if summary:
                        lines.append(f"  {summary}")
            lines.append("")
    else:
        lines.append("No enriched story threads were produced.")
        lines.append("")
    warnings = payload.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.append("## Warnings")
        for warning in warnings:
            text = str(warning or "").strip()
            if text:
                lines.append(f"- {text}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _refetch_degraded_article_texts(orchestrator, articles: List[SelectedArticle], warnings: List[str]) -> None:
    degraded = [
        article
        for article in articles
        if article.candidate.url
        and str(article.extraction_status or "").startswith("degraded")
    ]
    if not degraded:
        return
    max_chars = max(
        int(getattr(orchestrator.config.general_filtering, "article_text_max_chars", 0) or 0),
        int(getattr(orchestrator.config.filtering, "article_text_max_chars", 0) or 0),
        3000,
    )
    retriever = ArticleRetriever(
        orchestrator.config.user_agent,
        max_chars,
        http_cache=None,
        debug=orchestrator.debug,
    )
    populate_article_texts(
        brief_name="enrichment_module",
        selected=degraded,
        article_retriever=retriever,
        warnings=warnings,
        max_article_workers=orchestrator.config.runtime.max_article_workers,
        debug=orchestrator.debug,
        article_text_cache=getattr(orchestrator, "article_text_cache", None),
    )
    for article in degraded:
        if not article.article_text:
            article.article_text = article.candidate.snippet or article.candidate.title
            article.extraction_status = "degraded_brief_json"


def _record_enrichment_artifact(orchestrator, *, payload: Dict[str, Any], markdown_path: Path, json_path: Path) -> None:
    stage_payload_builder = getattr(orchestrator, "_stage_payload", None)
    record_stage_artifact = getattr(orchestrator, "_record_stage_artifact", None)
    if not callable(stage_payload_builder) or not callable(record_stage_artifact):
        return
    record_stage_artifact(
        stage="enrichment",
        brief_name="pipeline",
        payload=stage_payload_builder(
            stage="enrichment",
            brief_name="pipeline",
            summary={
                "source_briefs": payload.get("source_briefs", []),
                "selected": len(payload.get("selected_articles", [])),
                "story_threads": len(payload.get("story_threads", [])),
                "markdown_path": str(markdown_path),
                "json_path": str(json_path),
                "warnings": len(payload.get("warnings", [])),
            },
            next_stage_input={
                "enrichment": payload,
                "markdown_path": str(markdown_path),
                "json_path": str(json_path),
            },
        ),
    )


def _dedupe_articles(articles: List[SelectedArticle]) -> List[SelectedArticle]:
    by_key: Dict[str, SelectedArticle] = {}
    for article in articles:
        key = _article_dedupe_key(article)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = article
            continue
        winner = _preferred_article(existing, article)
        loser = article if winner is existing else existing
        _merge_article_source_metadata(winner, loser)
        by_key[key] = winner
    return list(by_key.values())


def _article_dedupe_key(article: SelectedArticle) -> str:
    normalized = normalize_url(article.candidate.url)
    if normalized:
        return f"url:{normalized}"
    return f"id:{article.candidate.id}"


def _preferred_article(left: SelectedArticle, right: SelectedArticle) -> SelectedArticle:
    left_mode = str(left.candidate.metadata.get("input_mode", "") or "")
    right_mode = str(right.candidate.metadata.get("input_mode", "") or "")
    if left_mode != right_mode:
        if right_mode == "handoff":
            return right
        if left_mode == "handoff":
            return left
    if len(right.article_text or "") > len(left.article_text or ""):
        return right
    return left


def _merge_article_source_metadata(target: SelectedArticle, other: SelectedArticle) -> None:
    metadata = target.candidate.metadata
    for key in ("source_briefs", "source_json_paths", "input_modes"):
        current = list(metadata.get(key, []) if isinstance(metadata.get(key), list) else [])
        other_values = other.candidate.metadata.get(key, [])
        if not isinstance(other_values, list):
            other_values = []
        for item in other_values:
            if item not in current:
                current.append(item)
        metadata[key] = current


def _record_article_source(article: SelectedArticle, brief_name: str, mode: str) -> None:
    metadata = article.candidate.metadata
    source_briefs = list(metadata.get("source_briefs", []) if isinstance(metadata.get("source_briefs"), list) else [])
    if brief_name and brief_name not in source_briefs:
        source_briefs.append(brief_name)
    metadata["source_briefs"] = source_briefs
    input_modes = list(metadata.get("input_modes", []) if isinstance(metadata.get("input_modes"), list) else [])
    if mode and mode not in input_modes:
        input_modes.append(mode)
    metadata["input_modes"] = input_modes
    metadata["input_mode"] = mode


def _article_source_briefs(article: SelectedArticle) -> List[str]:
    value = article.candidate.metadata.get("source_briefs", [])
    return [str(item) for item in value] if isinstance(value, list) else []


def _enrichment_enabled(config) -> bool:
    if not bool(getattr(config.enrichment, "enabled", False)):
        return False
    mode = str(getattr(config.enrichment, "mode", "story_llm") or "story_llm").strip().lower()
    return mode != "disabled"
