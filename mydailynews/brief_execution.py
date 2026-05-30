from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Dict, List

from .ai.headline_analyzer import HeadlineAnalyzer
from .brief import BriefGenerator, brief_metadata
from .enrichment import SimpleEnricher
from .models import (
    BriefOutput,
    HeadlineDecision,
    NewsCandidate,
    PriorReport,
    RunSourceSnapshot,
    TopicConfig,
)
from .output import write_json, write_markdown
from .retrieval.article import ArticleRetriever


def run_brief(
    orchestrator,
    *,
    name: str,
    output_suffix: str,
    topics: List[TopicConfig],
    filtering,
    prior_reports: List[PriorReport],
    now,
    date: str,
    snapshot: RunSourceSnapshot | None,
    brief_goal: str,
    limited_candidates_override: List[NewsCandidate] | None = None,
    shared_decisions: Dict[str, HeadlineDecision] | None = None,
) -> BriefOutput:
    with orchestrator.debug.span(f"brief.{name}.total"):
        since = now - timedelta(hours=filtering.time_window_hours)
        run_warnings: List[str] = []
        orchestrator.debug.set_metric(f"brief.{name}.status", "running")
        orchestrator.debug.log(
            "brief.run",
            "starting",
            name=name,
            topics=len(topics),
            max_candidates=filtering.max_candidates_for_ai,
            ai_batch_size=filtering.max_headlines_per_ai_batch,
            cutoff=filtering.headline_score_cutoff,
            max_selected=filtering.max_selected_articles,
            fill=filtering.fill_selected_articles,
        )

        try:
            unique_candidates: List[NewsCandidate]
            raw_candidate_count = 0
            rss_candidate_count = 0
            topic_candidate_count = 0
            with orchestrator.debug.span(f"brief.{name}.candidate_prepare"):
                if snapshot:
                    run_warnings.extend(str(item) for item in snapshot.metadata.get("warnings", []))
                    rss_candidates, topic_candidates, unique_candidates = orchestrator._snapshot_candidates_for_brief(snapshot, since)
                    raw_candidate_count = len(rss_candidates) + len(topic_candidates)
                    rss_candidate_count = len(rss_candidates)
                    topic_candidate_count = len(topic_candidates)
                    orchestrator.debug.log(
                        "headline.fetch",
                        "reused_snapshot",
                        brief=name,
                        snapshot_since=snapshot.fetched_since,
                        raw_candidates=raw_candidate_count,
                        rss_candidates=rss_candidate_count,
                        topic_candidates=topic_candidate_count,
                        unique_candidates=len(unique_candidates),
                        prior_reports=len(prior_reports),
                    )
                else:
                    with orchestrator.debug.span(f"brief.{name}.headline_fetch"):
                        candidates = orchestrator.fetch_headlines(since, filtering.max_headlines_per_source, run_warnings)
                        topic_candidates = orchestrator.fetch_topic_headlines(topics, since, run_warnings)
                    candidates.extend(topic_candidates)
                    raw_candidate_count = len(candidates)
                    rss_candidate_count = len(candidates) - len(topic_candidates)
                    topic_candidate_count = len(topic_candidates)
                    orchestrator.debug.log(
                        "headline.fetch",
                        "complete",
                        brief=name,
                        raw_candidates=raw_candidate_count,
                        rss_candidates=rss_candidate_count,
                        topic_candidates=topic_candidate_count,
                        prior_reports=len(prior_reports),
                    )
                    unique_candidates = orchestrator.merge_url_duplicates(candidates)
                    orchestrator.debug.log("headline.dedupe", "complete", brief=name, unique_candidates=len(unique_candidates))
            orchestrator.debug.set_metric(f"brief.{name}.raw_candidates", raw_candidate_count)
            orchestrator.debug.set_metric(f"brief.{name}.rss_candidates", rss_candidate_count)
            orchestrator.debug.set_metric(f"brief.{name}.topic_candidates", topic_candidate_count)
            orchestrator.debug.set_metric(f"brief.{name}.unique_candidates", len(unique_candidates))

            if not unique_candidates:
                run_warnings.append(f"{name}: No live headline candidates were fetched.")
            with orchestrator.debug.span(f"brief.{name}.headline_limit"):
                if limited_candidates_override is None:
                    limited_candidates = orchestrator.limit_candidates_for_ai(unique_candidates, topics, filtering, since)
                    orchestrator.debug.log("headline.limit", "complete", brief=name, candidates_for_ai=len(limited_candidates))
                else:
                    limited_candidates = list(limited_candidates_override)
                    orchestrator.debug.log("headline.limit", "reused_shared_prefilter", brief=name, candidates_for_ai=len(limited_candidates))
            orchestrator.debug.set_metric(f"brief.{name}.limited_candidates", len(limited_candidates))

            with orchestrator.debug.span(f"brief.{name}.headline_decisions"):
                if shared_decisions is None:
                    # Batch size is configurable; smaller values trade speed for reliability on constrained hardware.
                    headline_analyzer = HeadlineAnalyzer(
                        orchestrator.summary_ai_client,
                        max(1, int(filtering.max_headlines_per_ai_batch)),
                        orchestrator.debug,
                        cache=orchestrator.synth_cache,
                        cache_ttl_seconds=orchestrator.config.cache.synth_fresh_seconds,
                    )
                    decisions = headline_analyzer.analyze(
                        limited_candidates,
                        orchestrator.config.user_memory,
                        topics,
                        brief_goal,
                        brief_name=name,
                    )
                    run_warnings.extend(headline_analyzer.warnings)
                    orchestrator.debug.log("headline.decisions", "complete", brief=name, decisions=len(decisions))
                else:
                    decisions = orchestrator._decisions_for_brief(limited_candidates, shared_decisions, topics)
                    orchestrator.debug.log("headline.decisions", "reused_shared", brief=name, decisions=len(decisions))
            orchestrator.debug.set_metric(f"brief.{name}.decisions", len(decisions))

            with orchestrator.debug.span(f"brief.{name}.headline_select"):
                selected = orchestrator.select_articles(limited_candidates, decisions, topics, filtering)
            orchestrator.debug.set_metric(f"brief.{name}.selected", len(selected))
            orchestrator.debug.log("headline.select", "complete", brief=name, selected=len(selected))
            if not selected:
                orchestrator.debug.set_metric(f"brief.{name}.status", "failed")
                raise RuntimeError(
                    f"{name}: selected 0 articles from {len(limited_candidates)} scored candidates; "
                    "aborting before final synthesis. Check output/diagnostics for scorer failure artifacts."
                )

            article_retriever = ArticleRetriever(
                orchestrator.config.user_agent,
                filtering.article_text_max_chars,
                http_cache=orchestrator.http_cache,
                cache_fresh_seconds=orchestrator.config.cache.http_fresh_seconds,
                debug=orchestrator.debug,
            )
            enricher = SimpleEnricher(
                orchestrator.config,
                http_cache=orchestrator.http_cache,
                debug=orchestrator.debug,
            )
            for article in selected:
                orchestrator.debug.log(
                    "article",
                    "selected",
                    brief=name,
                    score=article.decision.score,
                    topic=article.decision.topic,
                    source=article.candidate.source,
                    title=article.candidate.title,
                )
            with orchestrator.debug.span(f"brief.{name}.article_fetch"):
                orchestrator._populate_article_texts(name, selected, article_retriever, run_warnings)
            orchestrator._record_article_fetch_metrics(name, selected)

            with orchestrator.debug.span(f"brief.{name}.enrichment"):
                enricher.enrich_many(selected, max_workers=orchestrator.config.runtime.max_enrichment_workers)
            orchestrator._record_enrichment_metrics(name, selected)

            if orchestrator.summary_ai_client is not orchestrator.final_ai_client:
                orchestrator.summary_ai_client.unload()

            brief_generator = BriefGenerator(
                orchestrator.final_ai_client,
                orchestrator.config.enrichment.max_context_chars_per_article,
                input_token_limit=orchestrator.config.ai_final.max_input_tokens,
                max_new_tokens=orchestrator.config.ai_final.max_new_tokens,
                debug=orchestrator.debug,
            )
            with orchestrator.debug.span(f"brief.{name}.final_brief"):
                brief = brief_generator.generate(
                    selected,
                    orchestrator.config.user_memory,
                    topics,
                    prior_reports,
                    brief_goal,
                    date,
                    brief_name=name,
                )
            orchestrator.final_ai_client.unload()
            run_warnings.extend(enricher.warnings)
            run_warnings.extend(brief_generator.warnings)
            brief["metadata"] = brief_metadata(
                date=date,
                model=f"{orchestrator.config.ai_summary.backend}:{orchestrator.config.ai_summary.effective_model_label} -> "
                f"{orchestrator.config.ai_final.backend}:{orchestrator.config.ai_final.effective_model_label}",
                candidate_count=len(unique_candidates),
                selected_count=len(selected),
                topics=[topic.name for topic in topics],
                prior_reports_count=len(prior_reports),
                brief_name=name,
                warnings=run_warnings,
            )

            output_dir = Path(orchestrator.config.output_dir)
            markdown_path = output_dir / f"{date}_{output_suffix}_brief.md"
            json_path = output_dir / f"{date}_{output_suffix}_brief.json"
            with orchestrator.debug.span(f"brief.{name}.write_output"):
                write_markdown(markdown_path, brief)
                write_json(json_path, brief)
            orchestrator.warnings.extend(run_warnings)
            orchestrator.debug.set_metric(f"brief.{name}.warnings", len(run_warnings))
            orchestrator.debug.set_metric(f"brief.{name}.status", "completed")
            orchestrator.debug.log("brief.run", "complete", name=name, markdown=markdown_path, json=json_path, warnings=len(run_warnings))

            return BriefOutput(
                name=name,
                markdown_path=str(markdown_path),
                json_path=str(json_path),
                candidate_count=len(unique_candidates),
                selected_count=len(selected),
                warnings=run_warnings,
            )
        except Exception as exc:
            orchestrator.debug.set_metric(f"brief.{name}.status", "failed")
            orchestrator.debug.set_metric(f"brief.{name}.error", f"{type(exc).__name__}: {exc}")
            raise
