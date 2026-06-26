from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from mydailynews.app.models import SelectedArticle
from mydailynews.story_grouping.models import ResearchQuestion, StoryGroup
from mydailynews.story_grouping.payloads import clean_text


QuestionParser = Callable[[Any, str], list[ResearchQuestion]]
FallbackQuestions = Callable[[str], list[ResearchQuestion]]


@dataclass
class StoryGroupNormalizationResult:
    groups: list[StoryGroup]
    warnings: list[str] = field(default_factory=list)
    unknown_article_ids: int = 0
    duplicate_article_ids: int = 0
    duplicate_story_ids: int = 0
    fallback_groups: int = 0


def normalize_story_groups(
    *,
    selected: list[SelectedArticle],
    raw_groups: Iterable[StoryGroup | dict[str, Any]],
    caller: str,
    allow_singleton_fallback: bool,
    fallback_questions: FallbackQuestions | None = None,
    question_parser: QuestionParser | None = None,
    fallback_when_empty_input: bool = True,
) -> StoryGroupNormalizationResult:
    """Normalize story-group-like objects against the selected article set."""

    raw_items = [item for item in raw_groups if isinstance(item, (StoryGroup, dict))]
    article_by_id = {article.candidate.id: article for article in selected}
    assigned: set[str] = set()
    story_ids: set[str] = set()
    groups: list[StoryGroup] = []
    warnings: list[str] = []
    unknown_article_ids = 0
    duplicate_article_ids = 0
    duplicate_story_ids = 0

    for raw in raw_items:
        raw_article_ids = _raw_value(raw, "article_ids", [])
        article_ids: list[str] = []
        if isinstance(raw_article_ids, list):
            for value in raw_article_ids:
                article_id = str(value or "").strip()
                if article_id not in article_by_id:
                    if article_id:
                        unknown_article_ids += 1
                        warnings.append(f"{caller} ignored unknown article id {article_id}")
                    continue
                if article_id in assigned:
                    duplicate_article_ids += 1
                    warnings.append(f"{caller} duplicate assignment ignored for article {article_id}")
                    continue
                assigned.add(article_id)
                article_ids.append(article_id)
        if not article_ids:
            continue

        story_id = clean_text(_raw_value(raw, "story_id", ""), 80) or _next_unused_story_id(
            len(groups) + 1,
            story_ids,
        )
        if story_id in story_ids:
            duplicate_story_ids += 1
            warnings.append(f"{caller} duplicate story id ignored: {story_id}")
            story_id = _next_unused_story_id(len(groups) + 1, story_ids)
        story_ids.add(story_id)

        story_title = clean_text(_raw_value(raw, "story_title", ""), 180) or _fallback_story_title(
            article_by_id,
            article_ids,
        )
        research_questions = _research_questions(raw, story_title, question_parser)
        groups.append(
            StoryGroup(
                story_id=story_id,
                story_title=story_title,
                article_ids=article_ids,
                research_questions=research_questions,
                fallback=bool(_raw_value(raw, "fallback", False)),
                topic=clean_text(_raw_value(raw, "topic", ""), 120),
            )
        )

    fallback_groups = 0
    omitted = [article for article in selected if article.candidate.id not in assigned]
    should_add_fallback = allow_singleton_fallback and omitted and (raw_items or fallback_when_empty_input)
    if should_add_fallback:
        warnings.append(
            f"{caller} omitted selected article(s); added singleton fallback group(s): "
            + ", ".join(article.candidate.id for article in omitted)
        )
        for article in omitted:
            story_title = clean_text(article.candidate.title, 180)
            story_id = _next_unused_story_id(len(groups) + 1, story_ids)
            story_ids.add(story_id)
            groups.append(
                StoryGroup(
                    story_id=story_id,
                    story_title=story_title,
                    article_ids=[article.candidate.id],
                    research_questions=fallback_questions(story_title) if fallback_questions else [],
                    fallback=True,
                    topic=clean_text(article.decision.topic or article.candidate.metadata.get("topic_name", ""), 120),
                )
            )
            fallback_groups += 1

    return StoryGroupNormalizationResult(
        groups=groups,
        warnings=warnings,
        unknown_article_ids=unknown_article_ids,
        duplicate_article_ids=duplicate_article_ids,
        duplicate_story_ids=duplicate_story_ids,
        fallback_groups=fallback_groups,
    )


def _raw_value(raw: StoryGroup | dict[str, Any], key: str, default: Any) -> Any:
    if isinstance(raw, StoryGroup):
        return getattr(raw, key, default)
    return raw.get(key, default)


def _research_questions(
    raw: StoryGroup | dict[str, Any],
    story_title: str,
    question_parser: QuestionParser | None,
) -> list[ResearchQuestion]:
    value = _raw_value(raw, "research_questions", [])
    if question_parser is not None:
        return question_parser(value, story_title)
    if not isinstance(value, list):
        return []
    questions: list[ResearchQuestion] = []
    for item in value:
        if isinstance(item, ResearchQuestion):
            questions.append(item)
        elif isinstance(item, dict):
            question = clean_text(item.get("question", ""), 240)
            raw_queries = item.get("queries", [])
            queries = [
                clean_text(query, 140)
                for query in (raw_queries if isinstance(raw_queries, list) else [raw_queries])
                if clean_text(query, 140)
            ][:3]
            if question or queries:
                questions.append(ResearchQuestion(question=question or queries[0], queries=queries or [question]))
    return questions


def _next_unused_story_id(start_index: int, used: set[str]) -> str:
    index = max(1, int(start_index))
    while True:
        story_id = f"story-{index:03d}"
        if story_id not in used:
            return story_id
        index += 1


def _fallback_story_title(article_by_id: dict[str, SelectedArticle], article_ids: list[str]) -> str:
    if not article_ids:
        return "Selected story group"
    first = article_by_id.get(article_ids[0])
    return clean_text(first.candidate.title if first else "Selected story group", 180)
