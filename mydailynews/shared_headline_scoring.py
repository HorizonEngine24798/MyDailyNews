from __future__ import annotations

from datetime import timedelta
from typing import Dict, List

from .ai.headline_analyzer import HeadlineAnalyzer
from .headline_selection import union_candidates_by_id
from .models import HeadlineDecision, NewsCandidate, RunSourceSnapshot, TopicConfig
from .snapshot_helpers import merge_topics_for_snapshot


def _max_optional_int(*values: int | None) -> int | None:
    present = [int(value) for value in values if value is not None]
    return max(present) if present else None


def score_snapshot_headlines_once(
    *,
    snapshot: RunSourceSnapshot,
    now,
    general_topics: List[TopicConfig],
    detailed_topics: List[TopicConfig],
    config,
    debug,
    summary_ai_client,
    synth_cache,
    limit_candidates_for_ai,
    snapshot_candidates_for_brief,
    analyzer_cls=HeadlineAnalyzer,
) -> tuple[Dict[str, List[NewsCandidate]], Dict[str, HeadlineDecision], List[str]]:
    with debug.span("headline.shared.total"):
        general_since = now - timedelta(hours=config.general_filtering.time_window_hours)
        detailed_since = now - timedelta(hours=config.filtering.time_window_hours)
        _, _, general_candidates = snapshot_candidates_for_brief(snapshot, general_since)
        _, _, detailed_candidates = snapshot_candidates_for_brief(snapshot, detailed_since)
        candidates_by_brief = {
            "general": limit_candidates_for_ai(
                general_candidates,
                general_topics,
                config.general_filtering,
                general_since,
            ),
            "detailed": limit_candidates_for_ai(
                detailed_candidates,
                detailed_topics,
                config.filtering,
                detailed_since,
            ),
        }
        shared_candidates = union_candidates_by_id(
            candidates_by_brief["general"],
            candidates_by_brief["detailed"],
        )
        batch_sizes = [
            max(1, int(config.general_filtering.max_headlines_per_ai_batch)),
            max(1, int(config.filtering.max_headlines_per_ai_batch)),
        ]
        debug.set_metric("headline.shared.general_candidates", len(candidates_by_brief["general"]))
        debug.set_metric("headline.shared.detailed_candidates", len(candidates_by_brief["detailed"]))
        debug.set_metric("headline.shared.union_candidates", len(shared_candidates))
        debug.set_metric("headline.shared.batch_size", min(batch_sizes))
        debug.log(
            "headline.shared",
            "prepared",
            general_candidates=len(candidates_by_brief["general"]),
            detailed_candidates=len(candidates_by_brief["detailed"]),
            union_candidates=len(shared_candidates),
            batch_size=min(batch_sizes),
        )
        if not shared_candidates:
            debug.set_metric("headline.shared.decisions", 0)
            return candidates_by_brief, {}, []

        headline_analyzer = analyzer_cls(
            summary_ai_client,
            min(batch_sizes),
            debug,
            cache=synth_cache,
            cache_ttl_seconds=config.cache.synth_fresh_seconds,
            input_token_limit=_max_optional_int(
                getattr(config.general_filtering, "headline_max_input_tokens", None),
                getattr(config.filtering, "headline_max_input_tokens", None),
            ),
            max_new_tokens=_max_optional_int(
                getattr(config.general_filtering, "headline_max_new_tokens", None),
                getattr(config.filtering, "headline_max_new_tokens", None),
            ),
            single_replay_max_new_tokens=_max_optional_int(
                getattr(config.general_filtering, "headline_single_replay_max_new_tokens", None),
                getattr(config.filtering, "headline_single_replay_max_new_tokens", None),
            ),
        )
        shared_topics = merge_topics_for_snapshot(general_topics, detailed_topics)
        shared_goal = (
            "Shared headline scoring pass for both brief modes. Score each candidate for usefulness either to the "
            "general daily brief or to the detailed topic brief. Favor important, relevant, fresh, high-signal "
            "stories that are worth retrieving and reading in full."
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
