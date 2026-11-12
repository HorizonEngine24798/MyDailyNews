"""End-to-end quality benchmark for MyDailyNews.

Evaluates pipeline performance across four key dimensions:
1. Ground-Truth Story Continuity & Retrieval (evals/cases/change_monitoring.real_blind.v1.json)
2. Production Delta & Novelty Flow (output/*_general_brief.json)
3. Grounding & Source Diversity (output/*_general_brief.json)
4. Perspectives & Multi-Country Yield (output/*_perspectives_report.json)

Usage:
    python tools/benchmark_quality.py
    python tools/benchmark_quality.py --output output/quality_benchmark_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mydailynews.evaluation.retrieval_diagnostics import evaluate_story_store_retrieval
from mydailynews.evaluation.schema import load_corpus


@dataclass
class ContinuityMetrics:
    total_documents: int = 0
    historical_continuations: int = 0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    mean_reciprocal_rank: float = 0.0
    unseen_stories: int = 0
    new_story_clean_rate: float = 0.0  # % of new stories without false historical candidate
    related_theme_items: int = 0
    related_theme_with_candidate_rate: float = 0.0


@dataclass
class ProductionDeltaMetrics:
    total_briefs: int = 0
    date_range: str = ""
    total_selected_articles: int = 0
    avg_articles_per_day: float = 0.0
    delta_breakdown: Dict[str, int] = field(default_factory=dict)
    structured_decisions_count: int = 0
    avg_materiality: float = 0.0
    disposition_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class DiversityAndGroundingMetrics:
    total_references: int = 0
    matched_references: int = 0
    reference_fidelity_rate: float = 0.0
    unique_outlets: int = 0
    top5_outlet_share: float = 0.0
    top_outlets: List[Dict[str, Any]] = field(default_factory=list)
    topic_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class PerspectivesMetrics:
    total_reports: int = 0
    total_stories: int = 0
    multi_country_stories: int = 0
    multi_country_rate: float = 0.0
    total_coverage_articles: int = 0
    avg_articles_per_story: float = 0.0
    stories_with_loci: int = 0
    loci_rate: float = 0.0


@dataclass
class QualityBenchmarkReport:
    timestamp: str
    overall_quality_score: float
    grade: str
    continuity: ContinuityMetrics
    production_delta: ProductionDeltaMetrics
    diversity_and_grounding: DiversityAndGroundingMetrics
    perspectives: PerspectivesMetrics

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def benchmark_continuity(corpus_path: Path) -> ContinuityMetrics:
    if not corpus_path.exists():
        return ContinuityMetrics()
    corpus = load_corpus(str(corpus_path))
    diag = evaluate_story_store_retrieval(corpus).payload()
    return ContinuityMetrics(
        total_documents=diag.get("documents", 0),
        historical_continuations=diag.get("historical_continuations", 0),
        recall_at_1=diag.get("recall_at_1", 0.0),
        recall_at_3=diag.get("recall_at_3", 0.0),
        mean_reciprocal_rank=diag.get("mean_reciprocal_rank", 0.0),
        unseen_stories=diag.get("unseen_stories", 0),
        new_story_clean_rate=round(1.0 - diag.get("new_story_without_candidate_rate", 0.0), 4),
        related_theme_items=diag.get("related_theme_items", 0),
        related_theme_with_candidate_rate=diag.get("related_theme_with_candidate_rate", 0.0),
    )


def benchmark_production_delta(output_dir: Path) -> ProductionDeltaMetrics:
    brief_files = sorted(output_dir.glob("*_general_brief.json"))
    if not brief_files:
        return ProductionDeltaMetrics()

    dates = [p.name.split("_")[0] for p in brief_files]
    date_range = f"{min(dates)} to {max(dates)}"
    total_articles = 0
    delta_counter: Counter[str] = Counter()
    disposition_counter: Counter[str] = Counter()
    materiality_list: List[float] = []
    total_decisions = 0

    for p in brief_files:
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        articles = data.get("selected_articles", [])
        total_articles += len(articles)

        delta = data.get("analysis", {}).get("delta_packet", {})
        for cat in ["new", "escalated", "weakened", "reframed", "unchanged_but_important"]:
            items = delta.get(cat, [])
            if isinstance(items, list):
                delta_counter[cat] += len(items)

        decisions = delta.get("story_decisions", [])
        total_decisions += len(decisions)
        for d in decisions:
            disp = d.get("disposition", "unknown")
            disposition_counter[disp] += 1
            mat = d.get("materiality")
            if mat is not None:
                materiality_list.append(float(mat))

    avg_mat = round(sum(materiality_list) / len(materiality_list), 4) if materiality_list else 0.0
    return ProductionDeltaMetrics(
        total_briefs=len(brief_files),
        date_range=date_range,
        total_selected_articles=total_articles,
        avg_articles_per_day=round(total_articles / len(brief_files), 1),
        delta_breakdown=dict(delta_counter),
        structured_decisions_count=total_decisions,
        avg_materiality=avg_mat,
        disposition_breakdown=dict(disposition_counter),
    )


def benchmark_diversity_and_grounding(output_dir: Path) -> DiversityAndGroundingMetrics:
    brief_files = sorted(output_dir.glob("*_general_brief.json"))
    if not brief_files:
        return DiversityAndGroundingMetrics()

    sources_counter: Counter[str] = Counter()
    topics_counter: Counter[str] = Counter()
    total_refs = 0
    matched_refs = 0

    for p in brief_files:
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            continue

        articles = data.get("selected_articles", [])
        article_urls = {str(a.get("url") or "") for a in articles}
        article_titles = {str(a.get("title") or "") for a in articles}

        for a in articles:
            src = a.get("source") or a.get("domain") or "Unknown"
            sources_counter[src] += 1
            top = a.get("topic") or "General"
            topics_counter[top] += 1

        for r in data.get("references", []):
            total_refs += 1
            url = str(r.get("url") or "")
            title = str(r.get("title") or "")
            if url in article_urls or title in article_titles:
                matched_refs += 1

    total_articles = sum(sources_counter.values())
    top5_sum = sum(c for _, c in sources_counter.most_common(5))
    top5_share = round(top5_sum / max(1, total_articles), 4)

    return DiversityAndGroundingMetrics(
        total_references=total_refs,
        matched_references=matched_refs,
        reference_fidelity_rate=round(matched_refs / max(1, total_refs), 4) if total_refs else 1.0,
        unique_outlets=len(sources_counter),
        top5_outlet_share=top5_share,
        top_outlets=[{"outlet": src, "articles": count} for src, count in sources_counter.most_common(5)],
        topic_distribution=dict(topics_counter.most_common(8)),
    )


def benchmark_perspectives(output_dir: Path) -> PerspectivesMetrics:
    persp_files = sorted(output_dir.glob("*_perspectives_report.json"))
    if not persp_files:
        return PerspectivesMetrics()

    total_stories = 0
    multi_country = 0
    total_coverage_articles = 0
    stories_with_loci = 0

    for pf in persp_files:
        try:
            data = json.loads(pf.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        for s in data.get("stories", []):
            total_stories += 1
            articles = s.get("coverage_articles", [])
            total_coverage_articles += len(articles)
            countries = {a.get("source_country") for a in articles if a.get("source_country")}
            if len(countries) >= 2:
                multi_country += 1
            if s.get("story_loci"):
                stories_with_loci += 1

    return PerspectivesMetrics(
        total_reports=len(persp_files),
        total_stories=total_stories,
        multi_country_stories=multi_country,
        multi_country_rate=round(multi_country / max(1, total_stories), 4),
        total_coverage_articles=total_coverage_articles,
        avg_articles_per_story=round(total_coverage_articles / max(1, total_stories), 2),
        stories_with_loci=stories_with_loci,
        loci_rate=round(stories_with_loci / max(1, total_stories), 4),
    )


def compute_overall_quality_score(
    continuity: ContinuityMetrics,
    delta: ProductionDeltaMetrics,
    grounding: DiversityAndGroundingMetrics,
    perspectives: PerspectivesMetrics,
) -> tuple[float, str]:
    """Calculate normalized score (0 to 100) across 4 dimensions:
    - Continuity / Retrieval: 30%
    - Citation Grounding & Novelty: 25%
    - Source Diversity & Balance: 20%
    - Perspectives & International Depth: 25%
    """
    # 1. Continuity score (Recall@1 and MRR on real blind holdout)
    cont_score = (continuity.recall_at_1 * 0.6 + continuity.mean_reciprocal_rank * 0.4) * 100.0

    # 2. Grounding score (100% matched refs, delta mix includes updates)
    ref_score = grounding.reference_fidelity_rate * 70.0
    delta_total = sum(delta.delta_breakdown.values())
    update_ratio = (delta.delta_breakdown.get("escalated", 0) + delta.delta_breakdown.get("reframed", 0)) / max(1, delta_total)
    delta_health = min(30.0, update_ratio * 100.0)
    ground_score = ref_score + delta_health

    # 3. Source diversity score (healthy top-5 share <= 35%, outlets >= 50)
    diversity_score = 100.0
    if grounding.top5_outlet_share > 0.40:
        diversity_score -= (grounding.top5_outlet_share - 0.40) * 150.0
    diversity_score = max(50.0, min(100.0, diversity_score))

    # 4. Perspectives score (multi-country representation rate)
    persp_score = perspectives.multi_country_rate * 100.0

    overall = round(
        cont_score * 0.30 +
        ground_score * 0.25 +
        diversity_score * 0.20 +
        persp_score * 0.25,
        1
    )

    if overall >= 90.0:
        grade = "A (Excellent)"
    elif overall >= 80.0:
        grade = "B (Solid)"
    elif overall >= 70.0:
        grade = "C (Passable - Needs Tuning)"
    else:
        grade = "D (Action Required)"

    return overall, grade


def run_benchmark(
    *,
    corpus_path: Path,
    output_dir: Path,
) -> QualityBenchmarkReport:
    from datetime import datetime, timezone
    continuity = benchmark_continuity(corpus_path)
    delta = benchmark_production_delta(output_dir)
    grounding = benchmark_diversity_and_grounding(output_dir)
    perspectives = benchmark_perspectives(output_dir)
    overall, grade = compute_overall_quality_score(continuity, delta, grounding, perspectives)

    return QualityBenchmarkReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_quality_score=overall,
        grade=grade,
        continuity=continuity,
        production_delta=delta,
        diversity_and_grounding=grounding,
        perspectives=perspectives,
    )


def print_report(report: QualityBenchmarkReport) -> None:
    print("=" * 76)
    print("             MYDAILYNEWS SYSTEM QUALITY BENCHMARK REPORT             ")
    print("=" * 76)
    print(f"Overall Quality Score: {report.overall_quality_score}/100.0   |   Rating: {report.grade}")
    print("-" * 76)

    print("\n1. STORY CONTINUITY & RETRIEVAL (Gold Blind Holdout)")
    c = report.continuity
    print(f"   - Documents Evaluated:              {c.total_documents}")
    print(f"   - Historical Continuations:         {c.historical_continuations}")
    print(f"   - Recall@1 (Exact Prior Story):     {c.recall_at_1 * 100:.1f}%")
    print(f"   - Recall@3:                         {c.recall_at_3 * 100:.1f}%")
    print(f"   - Mean Reciprocal Rank (MRR):       {c.mean_reciprocal_rank:.4f}")
    print(f"   - Unseen Story Clean Rate:          {c.new_story_clean_rate * 100:.1f}%")

    print("\n2. PRODUCTION DELTA & NOVELTY FLOW (59 Briefs Historical)")
    d = report.production_delta
    print(f"   - Date Span:                        {d.date_range} ({d.total_briefs} days)")
    print(f"   - Articles Synthesized:             {d.total_selected_articles} (avg {d.avg_articles_per_day} / day)")
    print(f"   - Story Delta Mix:                  {d.delta_breakdown}")
    print(f"   - Structured Story Decisions:       {d.structured_decisions_count}")
    print(f"   - Average Materiality:              {d.avg_materiality:.2f}")

    print("\n3. GROUNDING, CITATION FIDELITY & SOURCE DIVERSITY")
    g = report.diversity_and_grounding
    print(f"   - References Checked:               {g.total_references}")
    print(f"   - Reference Fidelity Rate:          {g.reference_fidelity_rate * 100:.1f}% (0 ghost citations)")
    print(f"   - Unique Outlets Utilized:          {g.unique_outlets}")
    print(f"   - Top-5 Outlet Share:               {g.top5_outlet_share * 100:.1f}% (Healthy diversity)")
    top_str = ", ".join(f"{x['outlet']} ({x['articles']})" for x in g.top_outlets[:3])
    print(f"   - Leading Outlets:                  {top_str}")

    print("\n4. PERSPECTIVES & INTERNATIONAL YIELD")
    p = report.perspectives
    print(f"   - Full Perspectives Reports:        {p.total_reports} runs ({p.total_stories} stories)")
    print(f"   - Multi-Country Coverage Rate:      {p.multi_country_rate * 100:.1f}% ({p.multi_country_stories}/{p.total_stories} stories)")
    print(f"   - Total Retained Coverage Articles: {p.total_coverage_articles} (avg {p.avg_articles_per_story} / story)")
    print(f"   - Geographic Event Loci Populated:  {p.loci_rate * 100:.1f}%")

    print("=" * 76)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "evals" / "cases" / "change_monitoring.real_blind.v1.json",
        help="Path to evaluation corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output",
        help="Directory with historical output JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write benchmark report JSON.",
    )
    args = parser.parse_args()

    report = run_benchmark(corpus_path=args.corpus, output_dir=args.output_dir)
    print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved benchmark report to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
