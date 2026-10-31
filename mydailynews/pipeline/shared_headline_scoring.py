from __future__ import annotations

from datetime import timedelta
from typing import Dict, List, Sequence

from mydailynews.ai.headline_analyzer import HeadlineAnalyzer
from mydailynews.domain.headline_selection import limit_candidates_for_ai, union_candidates_by_id
from mydailynews.app.models import HeadlineDecision, NewsCandidate, RunSourceSnapshot, TopicConfig
from mydailynews.pipeline.brief_specs import BriefSpec
from mydailynews.pipeline.snapshot_helpers import merge_topics_for_snapshot, snapshot_candidates_for_brief


def _max_optional_int(*values: int | None) -> int | None:
    present = [int(value) for value in values if value is not None]
    return max(present) if present else None


def score_snapshot_headlines_once(
    *,
    snapshot: RunSourceSnapshot,
    now,
    briefs: Sequence[BriefSpec],
    config,
    debug,
    summary_ai_client,
    synth_cache,
    analyzer_cls=HeadlineAnalyzer,
) -> tuple[Dict[str, List[NewsCandidate]], Dict[str, HeadlineDecision], List[str]]:
    with debug.span("headline.shared.total"):
        candidates_by_brief: Dict[str, List[NewsCandidate]] = {}
        for brief in briefs:
            since = now - timedelta(hours=brief.filtering.time_window_hours)
            _, _, candidates = snapshot_candidates_for_brief(snapshot, since)
            candidates_by_brief[brief.name] = limit_candidates_for_ai(
                candidates,
                brief.topics,
                brief.filtering,
                since,
                user_memory=config.user_memory,
                debug=debug,
            )
        shared_candidates = union_candidates_by_id(*(candidates_by_brief.values()))
        batch_sizes = [max(1, int(brief.filtering.max_headlines_per_ai_batch)) for brief in briefs]
        for name, candidates in candidates_by_brief.items():
            debug.set_metric(f"headline.shared.{name}_candidates", len(candidates))
        debug.set_metric("headline.shared.union_candidates", len(shared_candidates))
        debug.set_metric("headline.shared.batch_size", min(batch_sizes, default=1))
        debug.log(
            "headline.shared",
            "prepared",
            union_candidates=len(shared_candidates),
            batch_size=min(batch_sizes, default=1),
            briefs=len(briefs),
        )
        if not shared_candidates:
            debug.set_metric("headline.shared.decisions", 0)
            return candidates_by_brief, {}, []

        headline_analyzer = analyzer_cls(
            summary_ai_client,
            min(batch_sizes, default=1),
            debug,
            cache=synth_cache,
            cache_ttl_seconds=config.cache.synth_fresh_seconds,
            input_token_limit=_max_optional_int(
                *(getattr(brief.filtering, "headline_max_input_tokens", None) for brief in briefs),
            ),
            max_new_tokens=_max_optional_int(
                *(getattr(brief.filtering, "headline_max_new_tokens", None) for brief in briefs),
            ),
            single_replay_max_new_tokens=_max_optional_int(
                *(getattr(brief.filtering, "headline_single_replay_max_new_tokens", None) for brief in briefs),
            ),
        )
        shared_topics = merge_topics_for_snapshot(*(brief.topics for brief in briefs))
        shared_goal = "Shared headline scoring pass. A candidate is useful when it serves any configured brief goal:\n" + "\n".join(
            f"- {brief.name}: {brief.goal}" for brief in briefs
        )
        decisions = headline_analyzer.analyze(
            shared_candidates,
            config.user_memory,
            shared_topics,
            shared_goal,
            brief_name="shared",
        )
        debug.set_metric("headline.shared.decisions", len(decisions))
        debug.log(
            "headline.shared",
            "complete",
            union_candidates=len(shared_candidates),
            decisions=len(decisions),
            warnings=len(headline_analyzer.warnings),
        )
        return candidates_by_brief, decisions, headline_analyzer.warnings
