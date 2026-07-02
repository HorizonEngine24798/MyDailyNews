from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, List

from mydailynews.app.models import NewsCandidate


WEAK_TERMS = {
    "breaking",
    "briefing",
    "daily",
    "developing",
    "headline",
    "headlines",
    "latest",
    "live",
    "news",
    "today",
    "update",
    "updates",
    "watch",
}

STOPWORDS = WEAK_TERMS.union(
    {
        "about",
        "after",
        "again",
        "against",
        "also",
        "amid",
        "among",
        "and",
        "are",
        "around",
        "before",
        "but",
        "can",
        "could",
        "from",
        "has",
        "have",
        "her",
        "his",
        "how",
        "into",
        "its",
        "may",
        "more",
        "new",
        "not",
        "over",
        "says",
        "say",
        "the",
        "their",
        "this",
        "through",
        "under",
        "was",
        "what",
        "when",
        "where",
        "who",
        "why",
        "will",
        "with",
    }
)


@dataclass(frozen=True)
class StoryIdentity:
    story_key: str
    story_family_key: str
    story_title: str
    tokens: List[str]
    match_confidence: float = 1.0


def story_identity_for_candidate(candidate: NewsCandidate) -> StoryIdentity:
    tokens = story_tokens_for_candidate(candidate)
    title_tokens = _tokens(candidate.title)
    key_tokens = title_tokens[:6] if len(title_tokens) >= 3 else tokens[:6]
    story_key = slugify_tokens(key_tokens)
    if not story_key:
        story_key = _fallback_key(candidate)
    family_tokens = tokens[:2] if len(tokens) >= 2 else tokens
    family_key = slugify_tokens(family_tokens) or story_key
    title = _clean_title(candidate.title) or story_key.replace("-", " ").title()
    return StoryIdentity(
        story_key=story_key,
        story_family_key=family_key,
        story_title=title[:140],
        tokens=tokens[:16],
        match_confidence=1.0 if tokens else 0.35,
    )


def story_tokens_for_candidate(candidate: NewsCandidate) -> List[str]:
    title_tokens = _tokens(candidate.title)
    topic_tokens = _tokens(str(candidate.metadata.get("topic_name", "") or candidate.category or ""))
    tag_tokens = _tokens(" ".join(str(tag) for tag in (candidate.tags or [])[:8]))
    snippet_tokens = _tokens(candidate.snippet)
    source_tokens = _tokens(candidate.source)

    ordered: List[str] = []
    ordered.extend(title_tokens[:10])
    ordered.extend(topic_tokens[:4])
    ordered.extend(tag_tokens[:4])
    ordered.extend(snippet_tokens[:8])
    if len(ordered) < 4:
        ordered.extend(source_tokens[:2])
    return _dedupe(ordered)[:18]


def token_overlap_confidence(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {str(item) for item in left if str(item)}
    right_set = {str(item) for item in right if str(item)}
    if not left_set or not right_set:
        return 0.0
    overlap = len(left_set.intersection(right_set))
    if overlap <= 0:
        return 0.0
    containment = overlap / max(1, min(len(left_set), len(right_set)))
    jaccard = overlap / max(1, len(left_set.union(right_set)))
    return round(max(containment * 0.78, jaccard), 4)


def slugify_tokens(tokens: Iterable[str], *, max_tokens: int = 6) -> str:
    cleaned = []
    for token in tokens:
        normalized = re.sub(r"[^a-z0-9]+", "", str(token or "").lower())
        if not normalized:
            continue
        cleaned.append(normalized)
        if len(cleaned) >= max_tokens:
            break
    return "-".join(cleaned)


def slugify_text(text: str, *, max_tokens: int = 6) -> str:
    return slugify_tokens(_tokens(text), max_tokens=max_tokens)


def _tokens(text: Any) -> List[str]:
    raw = re.findall(r"[a-z0-9]{3,}", str(text or "").lower())
    return [token for token in raw if token not in STOPWORDS and not token.isdigit()]


def _dedupe(tokens: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        normalized = str(token or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _clean_title(title: str) -> str:
    text = " ".join(str(title or "").split()).strip()
    text = re.sub(r"^(breaking|live updates?|latest|news):\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _fallback_key(candidate: NewsCandidate) -> str:
    for value in (candidate.title, candidate.url, candidate.id, candidate.source):
        key = slugify_text(str(value or ""), max_tokens=6)
        if key:
            return key
    return "story"
