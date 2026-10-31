from __future__ import annotations

from dataclasses import dataclass
from typing import List

from mydailynews.app.models import FilteringConfig, TopicConfig


DEFAULT_BRIEF_NAMES = ("general", "detailed")


@dataclass(frozen=True)
class BriefSpec:
    name: str
    output_suffix: str
    goal: str
    topics: List[TopicConfig]
    filtering: FilteringConfig


def brief_specs_from_config(config) -> List[BriefSpec]:
    return [
        BriefSpec(
            name="general",
            output_suffix="general",
            goal=(
                "General daily news pass. Prefer breadth and usefulness over deep specialization. "
                "Use the lower threshold to fill the brief with the strongest general stories, up to the configured "
                "article count. Still avoid trivia, gossip, minor sports, and duplicate rewrites."
            ),
            topics=[topic for topic in config.general_topics if topic.enabled],
            filtering=config.general_filtering,
        ),
        BriefSpec(
            name="detailed",
            output_suffix="detailed",
            goal=(
                "Detailed topic investigation pass. Focus on the configured topics, identify major narratives, "
                "compare with prior reports, and select sources that can deepen, challenge, or reshape those narratives."
            ),
            topics=[topic for topic in config.topics_to_examine if topic.enabled],
            filtering=config.filtering,
        ),
    ]
