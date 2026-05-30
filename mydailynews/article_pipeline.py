from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from .models import SelectedArticle


def populate_article_texts(
    *,
    brief_name: str,
    selected: List[SelectedArticle],
    article_retriever,
    warnings: List[str],
    max_article_workers: int,
    debug,
) -> None:
    if not selected:
        return
    worker_count = min(max(1, int(max_article_workers)), len(selected))
    debug.log("article.fetch", "batch_start", brief=brief_name, selected=len(selected), workers=worker_count)

    if worker_count <= 1:
        for article in selected:
            article.article_text, article.extraction_status = article_retriever.fetch_text(article.candidate.url)
            if not article.article_text:
                article.article_text = article.candidate.snippet
        debug.log("article.fetch", "batch_complete", brief=brief_name, selected=len(selected), workers=worker_count)
        return

    results: Dict[int, Tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(article_retriever.fetch_text, article.candidate.url): index
            for index, article in enumerate(selected)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            article = selected[index]
            try:
                results[index] = future.result()
            except Exception as exc:
                warnings.append(f"article fetch {article.candidate.id}: worker_exception={type(exc).__name__}")
                debug.log(
                    "article.fetch",
                    "worker_exception",
                    brief=brief_name,
                    article_id=article.candidate.id,
                    error=type(exc).__name__,
                )
                results[index] = ("", "worker_exception")

    for index, article in enumerate(selected):
        text, status = results.get(index, ("", "worker_missing"))
        article.article_text = text or article.candidate.snippet
        article.extraction_status = status
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
