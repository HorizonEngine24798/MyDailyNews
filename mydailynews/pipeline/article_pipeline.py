from __future__ import annotations

from typing import Dict, List, Tuple

from mydailynews.app.models import SelectedArticle
from mydailynews.common.parallel import ordered_parallel_map
from mydailynews.domain.article_identity import article_aliases_for_candidate
from mydailynews.retrieval.article_cache import ArticleTextCache


def _apply_article_fetch_result(article: SelectedArticle, text: str, status: str, resolved_url: str) -> None:
    original_url = article.candidate.url
    if resolved_url and resolved_url != original_url:
        article.candidate.metadata.setdefault("original_url", original_url)
        article.candidate.metadata["resolved_url"] = resolved_url
        article.candidate.url = resolved_url
    article.article_text = text or article.candidate.snippet
    article.extraction_status = status


def populate_article_texts(
    *,
    brief_name: str,
    selected: List[SelectedArticle],
    article_retriever,
    warnings: List[str],
    max_article_workers: int,
    debug,
    article_text_cache: ArticleTextCache | None = None,
) -> None:
    if not selected:
        return
    worker_count = min(max(1, int(max_article_workers)), len(selected))
    debug.log("article.fetch", "batch_start", brief=brief_name, selected=len(selected), workers=worker_count)

    def fetch_article(article: SelectedArticle) -> Tuple[str, str, str]:
        aliases = article_aliases_for_candidate(article.candidate)
        cached = article_text_cache.get_by_aliases(aliases) if article_text_cache else None
        if cached is not None:
            return (
                str(cached.get("article_text", "")),
                str(cached.get("extraction_status", "ok") or "ok"),
                str(cached.get("resolved_url") or cached.get("url") or article.candidate.url),
            )
        text, status, resolved_url = article_retriever.fetch_text_with_url(article.candidate.url)
        if article_text_cache:
            article_text_cache.store(
                candidate=article.candidate,
                aliases=aliases,
                article_text=text,
                extraction_status=status,
                resolved_url=resolved_url or article.candidate.url,
            )
        return text, status, resolved_url or article.candidate.url

    def handle_exception(_index: int, article: SelectedArticle, exc: Exception) -> Tuple[str, str, str]:
        warnings.append(f"article fetch {article.candidate.id}: worker_exception={type(exc).__name__}")
        debug.log(
            "article.fetch",
            "worker_exception",
            brief=brief_name,
            article_id=article.candidate.id,
            error=type(exc).__name__,
        )
        return "", "worker_exception", article.candidate.url

    results = ordered_parallel_map(
        selected,
        worker_count,
        fetch_article,
        on_exception=handle_exception,
    )
    for article, result in zip(selected, results):
        text, status, resolved_url = result
        _apply_article_fetch_result(article, text, status, resolved_url)
    debug.log("article.fetch", "batch_complete", brief=brief_name, selected=len(selected), workers=worker_count)


def record_article_fetch_metrics(
    *,
    brief_name: str,
    selected: List[SelectedArticle],
    debug,
) -> None:
    debug.set_metric(f"brief.{brief_name}.article_fetch.attempted", len(selected))
    status_counts: Dict[str, int] = {}
    for article in selected:
        status = article.extraction_status or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
    ok = status_counts.get("ok", 0)
    short_text = status_counts.get("short_text", 0)
    failed = len(selected) - ok - short_text
    debug.set_metric(f"brief.{brief_name}.article_fetch.ok", ok)
    debug.set_metric(f"brief.{brief_name}.article_fetch.short_text", short_text)
    debug.set_metric(f"brief.{brief_name}.article_fetch.failed", max(0, failed))
    debug.increment("article.fetch.attempted", len(selected))
    debug.increment("article.fetch.ok", ok)
    debug.increment("article.fetch.short_text", short_text)
    debug.increment("article.fetch.failed", max(0, failed))


def record_enrichment_metrics(
    *,
    brief_name: str,
    selected: List[SelectedArticle],
    debug,
) -> None:
    total = len(selected)
    needed = sum(1 for article in selected if article.enrichment_needed)
    skipped = total - needed
    wiki_results = sum(len(article.wikipedia_context) for article in selected)
    past_news_results = sum(len(article.past_news_context) for article in selected)
    context_sources = sum(len(article.context_sources) for article in selected)
    debug.set_metric(f"brief.{brief_name}.enrichment.total_articles", total)
    debug.set_metric(f"brief.{brief_name}.enrichment.needed", needed)
    debug.set_metric(f"brief.{brief_name}.enrichment.skipped", skipped)
    debug.set_metric(f"brief.{brief_name}.enrichment.wikipedia_results", wiki_results)
    debug.set_metric(f"brief.{brief_name}.enrichment.past_news_results", past_news_results)
    debug.set_metric(f"brief.{brief_name}.enrichment.context_sources", context_sources)
    debug.increment("enrichment.total_articles", total)
    debug.increment("enrichment.needed", needed)
    debug.increment("enrichment.skipped", skipped)
    debug.increment("enrichment.context_sources", context_sources)
