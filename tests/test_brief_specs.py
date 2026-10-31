from __future__ import annotations

import unittest
from contextlib import nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace

from mydailynews.app.models import FilteringConfig, RunSourceSnapshot, TopicConfig, UserMemory
from mydailynews.pipeline.brief_specs import BriefSpec
from mydailynews.pipeline.shared_headline_scoring import score_snapshot_headlines_once
from mydailynews.pipeline.snapshot_helpers import build_snapshot


class _Debug:
    def set_metric(self, *_args, **_kwargs) -> None:
        pass

    def log(self, *_args, **_kwargs) -> None:
        pass

    def span(self, *_args, **_kwargs):
        return nullcontext()


def _brief(name: str, hours: int, max_per_source: int) -> BriefSpec:
    return BriefSpec(
        name=name,
        output_suffix=name,
        goal=f"{name} goal",
        topics=[TopicConfig(name=name, queries=[name])],
        filtering=FilteringConfig(
            time_window_hours=hours,
            max_headlines_per_source=max_per_source,
            max_candidates_for_ai=10,
        ),
    )


class BriefSpecPipelineTests(unittest.TestCase):
    def test_snapshot_and_shared_scoring_accept_arbitrary_brief_specs(self) -> None:
        briefs = [_brief("first", 12, 3), _brief("second", 72, 11), _brief("third", 24, 5)]
        observed = {}

        def fetch_headlines(since, max_per_source, _warnings):
            observed["since"] = since
            observed["max_per_source"] = max_per_source
            return []

        snapshot = build_snapshot(
            use_shared_snapshot=True,
            now=datetime(2026, 9, 10, tzinfo=timezone.utc),
            briefs=briefs,
            debug=_Debug(),
            fetch_headlines=fetch_headlines,
            fetch_topic_headlines=lambda _topics, _since, _warnings: [],
            merge_url_duplicates=lambda items: items,
        )

        self.assertEqual(observed["since"], datetime(2026, 9, 7, tzinfo=timezone.utc))
        self.assertEqual(observed["max_per_source"], 11)
        self.assertEqual(snapshot.metadata["topic_count"], 3)

        candidates, decisions, warnings = score_snapshot_headlines_once(
            snapshot=RunSourceSnapshot(fetched_since=observed["since"]),
            now=datetime(2026, 9, 10, tzinfo=timezone.utc),
            briefs=briefs,
            config=SimpleNamespace(
                user_memory=UserMemory(),
                cache=SimpleNamespace(synth_fresh_seconds=0),
            ),
            debug=_Debug(),
            summary_ai_client=None,
            synth_cache=None,
        )
        self.assertEqual(set(candidates), {"first", "second", "third"})
        self.assertEqual(decisions, {})
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
