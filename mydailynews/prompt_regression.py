from __future__ import annotations

import json
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Dict, List

from .ai.headline_analyzer import HeadlineAnalyzer
from .analysis_pipeline import DeltaExtractor, EvidenceDistiller
from .brief import BriefGenerator
from .debug import DebugLogger
from .models import (
    ContextSource,
    DeltaExtractionConfig,
    EvidenceDistillationConfig,
    HeadlineDecision,
    NewsCandidate,
    PriorReport,
    SelectedArticle,
    TopicConfig,
    UserMemory,
)
from .utils import utc_now

PROMPT_REGRESSION_SCHEMA_VERSION = "prompt_regression_pack.v1"
PROMPT_REGRESSION_FIXTURE_VERSION = "2026-06-01"


class _PromptClient:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            backend="transformers",
            effective_model_label="prompt-regression-client",
            response_format="json_object",
        )
        self.max_input_tokens = 16384
        self.max_new_tokens = 1024

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def complete_json(*_args, **_kwargs):
        raise NotImplementedError("Prompt regression renderer does not execute model calls.")


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _stage_entry(prompt: str, required_clauses: List[str]) -> Dict[str, Any]:
    prompt_chars = len(prompt)
    return {
        "rendered_prompt": prompt,
        "prompt_chars": prompt_chars,
        "max_chars": max(prompt_chars + 400, int(prompt_chars * 1.25)),
        "max_char_delta": max(180, int(prompt_chars * 0.35)),
        "required_clauses": required_clauses,
    }


def _sample_context() -> Dict[str, Any]:
    now = utc_now()
    date = now.date().isoformat()
    memory = UserMemory(
        role="US policy-focused operator",
        geography_focus=["United States", "European Union"],
        time_horizon="strategic",
        beats={
            "AI policy": 1.0,
            "Semiconductor supply chain": 0.7,
            "Energy security": 0.5,
        },
        wants=["policy change", "regulatory enforcement", "supply chain risk"],
        avoid=["celebrity gossip", "live sports scores"],
        portfolio_or_stake_notes="Direct exposure to AI compute costs and export-control constraints.",
        preferred_depth="deep",
        briefing_style="Concise, skeptical, and action-oriented.",
    )
    topics = [
        TopicConfig(
            name="AI policy",
            description="US and EU AI governance, enforcement actions, and compliance timelines.",
            queries=["AI regulation", "AI Act enforcement", "model governance policy"],
        ),
        TopicConfig(
            name="Semiconductor supply chain",
            description="Export controls, fab capacity, advanced packaging, and critical suppliers.",
            queries=["chip export controls", "advanced packaging capacity", "fab supply chain"],
        ),
    ]
    prior_reports = [
        PriorReport(
            id="prior-1",
            date=(now - timedelta(days=1)).date().isoformat(),
            title="Daily Brief - Prior",
            path="output/prior.json",
            summary="Prior summary for AI policy and chip controls.",
            topics=["AI policy", "Semiconductor supply chain"],
            major_headlines=[
                {"headline": "US agency signals stricter AI model disclosure rules"},
                {"headline": "Chip equipment export rules tighten for advanced nodes"},
            ],
        ),
        PriorReport(
            id="prior-2",
            date=(now - timedelta(days=2)).date().isoformat(),
            title="Daily Brief - Earlier",
            path="output/prior_earlier.json",
            summary="Earlier baseline for supply and policy continuity.",
            topics=["AI policy"],
            major_headlines=[
                {"headline": "EU enforcement guidance narrows exemptions for foundation models"},
            ],
        ),
    ]
    candidate_a = NewsCandidate(
        id="cand-1",
        source="PolicyWire",
        category="policy",
        title="US agency advances enforceable AI disclosure rule with near-term compliance deadlines",
        url="https://example.com/cand-1",
        snippet="Draft text moves from principles to mandatory disclosures, raising immediate compliance planning needs.",
        published_at=now,
        metadata={
            "topic_name": "AI policy",
            "event_cluster_id": "evt-101",
            "event_cluster_label": "US AI disclosure enforcement",
            "event_cluster_size": 3,
            "event_cluster_source_count": 3,
            "event_cluster_multi_source": True,
            "event_cluster_latest_published_at": now.isoformat(),
        },
    )
    candidate_b = NewsCandidate(
        id="cand-2",
        source="SupplyWatch",
        category="industry",
        title="Advanced packaging bottlenecks persist as chip export controls broaden scope",
        url="https://example.com/cand-2",
        snippet="Suppliers report longer lead times as compliance screening expands across key equipment flows.",
        published_at=now - timedelta(hours=2),
        metadata={
            "topic_name": "Semiconductor supply chain",
            "event_cluster_id": "evt-202",
            "event_cluster_label": "Packaging + export control pressure",
            "event_cluster_size": 2,
            "event_cluster_source_count": 2,
            "event_cluster_multi_source": True,
            "event_cluster_latest_published_at": (now - timedelta(hours=2)).isoformat(),
        },
    )
    decision_a = HeadlineDecision(candidate_id="cand-1", score=8.9, topic="AI policy")
    decision_b = HeadlineDecision(candidate_id="cand-2", score=8.3, topic="Semiconductor supply chain")
    selected_a = SelectedArticle(
        candidate=candidate_a,
        decision=decision_a,
        article_text=(
            "Regulators published draft rule language requiring providers to disclose model risk controls, "
            "evaluation boundaries, and incident response timelines for high-impact use cases."
        ),
        extraction_status="ok",
        enrichment_reason="Cross-source policy continuity check.",
        context_sources=[
            ContextSource(
                id="ctx-1",
                parent_article_id="cand-1",
                kind="wikipedia_summary",
                title="AI governance",
                source="Wikipedia",
                url="https://wikipedia.org/wiki/AI_governance",
                summary="High-level governance context for AI model regulation.",
                items=[],
            )
        ],
    )
    selected_b = SelectedArticle(
        candidate=candidate_b,
        decision=decision_b,
        article_text=(
            "Packaging throughput remains constrained while export screening broadens, with suppliers signaling "
            "higher compliance overhead and delayed equipment shipments."
        ),
        extraction_status="ok",
        enrichment_reason="Supplier chain context support.",
        context_sources=[
            ContextSource(
                id="ctx-2",
                parent_article_id="cand-2",
                kind="past_news",
                title="Prior packaging constraints",
                source="PastNews",
                url="https://example.com/past-packaging",
                summary="Historical note on packaging capacity constraints.",
                items=[],
            )
        ],
    )
    evidence_packet = {
        "overview": "Policy enforcement language hardened while supply constraints remain unresolved.",
        "story_clusters": [
            {
                "cluster_id": "c1",
                "topic": "AI policy",
                "label": "Disclosure enforcement hardens",
                "summary": "Rule text now includes enforceable timelines.",
                "article_ids": ["cand-1"],
                "key_claims": [
                    {
                        "claim": "Disclosure deadlines are now explicit.",
                        "support_article_ids": ["cand-1"],
                        "confidence": "high",
                    }
                ],
                "consensus_points": ["Compliance workload likely rises in the near term."],
                "contested_points": ["Scope boundaries for lower-risk systems remain unclear."],
                "known_unknowns": ["Final enforcement start date is still pending."],
                "watch_signals": ["Publication date for final rule text."],
            }
        ],
        "global_watch_signals": ["Cross-agency alignment on model reporting templates."],
        "reader_qa": [
            {
                "question": "What should operators monitor first?",
                "answer": "Track final reporting schema and deadline cadence.",
                "article_ids": ["cand-1"],
            }
        ],
    }
    delta_packet = {
        "baseline_coverage_note": "Compared against two prior briefs with partial continuity.",
        "new": [
            {
                "item": "Enforceable disclosure deadlines",
                "summary": "Guidance moved from advisory framing to explicit compliance dates.",
                "article_ids": ["cand-1"],
            }
        ],
        "escalated": [
            {
                "item": "Packaging bottleneck pressure",
                "summary": "Lead-time risk remains elevated with broader export screening.",
                "article_ids": ["cand-2"],
            }
        ],
        "weakened": [],
        "reframed": [],
        "unchanged_but_important": [
            {
                "item": "Supplier concentration risk",
                "summary": "Dependency on a narrow supplier set still drives operational risk.",
                "article_ids": ["cand-2"],
            }
        ],
        "evidence_gaps": [
            {
                "gap": "No finalized regulator implementation FAQ yet.",
                "why_it_matters": "Execution burden and timelines remain uncertain.",
            }
        ],
    }
    return {
        "date": date,
        "brief_goal": "Detailed brief with decision-focused policy and supply-chain synthesis.",
        "memory": memory,
        "topics": topics,
        "prior_reports": prior_reports,
        "headline_candidates": [candidate_a, candidate_b],
        "selected_articles": [selected_a, selected_b],
        "evidence_packet": evidence_packet,
        "delta_packet": delta_packet,
    }


def response_shape_fixtures() -> Dict[str, Any]:
    return {
        "headline_analysis": {
            "decisions": [
                {
                    "id": "cand-1",
                    "score": 8.8,
                    "personal_relevance": 9.1,
                    "impact": 8.4,
                    "novelty": 7.2,
                    "urgency": 8.0,
                    "actionability": 8.5,
                    "confidence": 7.8,
                    "reason": "Direct policy change with immediate operational implications.",
                    "skip_reason": None,
                    "angle_type": "policy_change",
                },
                {
                    "id": "cand-2",
                    "score": 6.4,
                },
            ]
        },
        "evidence_distillation": {
            "overview": "Evidence indicates higher near-term compliance and supply pressure.",
            "story_clusters": [
                {
                    "cluster_id": "c1",
                    "topic": "AI policy",
                    "label": "Disclosure enforcement hardens",
                    "summary": "Mandatory deadlines are now explicit.",
                    "article_ids": ["cand-1"],
                    "key_claims": [
                        {
                            "claim": "Disclosure deadlines are explicit.",
                            "support_article_ids": ["cand-1"],
                            "confidence": "high",
                        }
                    ],
                    "consensus_points": ["Compliance workload likely rises."],
                    "contested_points": [],
                    "known_unknowns": ["Final implementation FAQ timing."],
                    "watch_signals": ["Regulator publication date."],
                }
            ],
            "global_watch_signals": ["Cross-agency template harmonization."],
            "reader_qa": [
                {
                    "question": "What should teams watch first?",
                    "answer": "Final reporting schema and deadline details.",
                    "article_ids": ["cand-1"],
                }
            ],
        },
        "delta_extraction": {
            "baseline_coverage_note": "Coverage continuity is moderate across two prior reports.",
            "new": [
                {
                    "item": "Enforceable disclosure deadlines",
                    "summary": "Language shifted to explicit compliance windows.",
                    "article_ids": ["cand-1"],
                }
            ],
            "escalated": [],
            "weakened": [],
            "reframed": [],
            "unchanged_but_important": [
                {
                    "item": "Packaging capacity risk",
                    "summary": "Constraint remains unresolved for advanced flows.",
                    "article_ids": ["cand-2"],
                }
            ],
            "evidence_gaps": [
                {
                    "gap": "Missing final implementation FAQ.",
                    "why_it_matters": "Operational planning remains approximate.",
                }
            ],
        },
        "final_brief": {
            "title": "Daily Brief - fixture",
            "lead": "Policy and supply signals tightened with near-term implications.",
            "topic_reports": [
                {
                    "topic": "AI policy",
                    "narrative_summary": "Guidance shifted toward enforceable disclosures.",
                    "narrative_changes": [
                        {
                            "narrative": "Disclosure enforcement",
                            "status": "escalating",
                            "summary": "Deadlines are now explicit for high-impact systems.",
                        }
                    ],
                    "what_to_watch": ["Final regulator publication date"],
                }
            ],
            "sections": [],
        },
    }


def build_prompt_regression_pack() -> Dict[str, Any]:
    data = _sample_context()
    memory = data["memory"]
    topics = data["topics"]
    brief_goal = data["brief_goal"]
    date = data["date"]
    candidates = data["headline_candidates"]
    selected_articles = data["selected_articles"]
    prior_reports = data["prior_reports"]
    evidence_packet = data["evidence_packet"]
    delta_packet = data["delta_packet"]

    client = _PromptClient()

    headline_analyzer = HeadlineAnalyzer(client, batch_size=4, debug=DebugLogger(False))
    headline_payload = [headline_analyzer._candidate_payload(item) for item in candidates]
    headline_prompt = headline_analyzer._build_user_prompt(memory, topics, brief_goal, headline_payload)

    evidence_distiller = EvidenceDistiller(
        client,
        EvidenceDistillationConfig(
            enabled=True,
            include_reader_qa=True,
            max_input_tokens=5200,
            max_new_tokens=640,
            max_articles=6,
            max_article_chars=700,
        ),
        include_enrichment_context=True,
        debug=DebugLogger(False),
    )
    evidence_prompt, _used_evidence_articles, _used_evidence_reports = evidence_distiller._build_prompt(
        articles=selected_articles,
        memory=memory,
        topics=topics,
        prior_reports=prior_reports,
        brief_goal=brief_goal,
        date=date,
    )

    delta_extractor = DeltaExtractor(
        client,
        DeltaExtractionConfig(
            enabled=True,
            input_source="evidence_or_articles",
            max_input_tokens=5200,
            max_new_tokens=420,
            max_prior_reports=3,
        ),
        debug=DebugLogger(False),
    )
    delta_prompt, _used_delta_articles, _used_delta_reports, _used_delta_evidence = delta_extractor._build_prompt(
        articles=selected_articles,
        memory=memory,
        topics=topics,
        prior_reports=prior_reports,
        brief_goal=brief_goal,
        date=date,
        evidence_packet=evidence_packet,
    )

    brief_generator = BriefGenerator(
        client,
        max_context_chars=720,
        input_token_limit=6400,
        max_new_tokens=900,
        include_enrichment_context=True,
        debug=DebugLogger(False),
    )
    final_prompt, _used_final_articles = brief_generator._build_prompt(
        selected_articles,
        memory,
        topics,
        prior_reports,
        brief_goal,
        date,
        evidence_packet=evidence_packet,
        delta_packet=delta_packet,
    )

    stages = {
        "headline_scoring": _stage_entry(
            headline_prompt,
            [
                "Use regret framing:",
                "Would this reader regret missing this today?",
                '"decisions"',
                '"score"',
            ],
        ),
        "evidence_distillation": _stage_entry(
            evidence_prompt,
            [
                "Work to perform:",
                "Cluster related developments into coherent story clusters.",
                '"story_clusters"',
                '"reader_qa"',
            ],
        ),
        "delta_extraction": _stage_entry(
            delta_prompt,
            [
                "Extract narrative deltas",
                "Identify what is new, escalated, weakened, reframed, or still important.",
                '"baseline_coverage_note"',
                '"evidence_gaps"',
            ],
        ),
        "final_brief_generation": _stage_entry(
            final_prompt,
            [
                "Synthesize only from the supplied article excerpts and context.",
                "Reject generic phrasing",
                '"why_it_matters"',
                '"what_to_watch"',
            ],
        ),
    }

    return {
        "schema_version": PROMPT_REGRESSION_SCHEMA_VERSION,
        "fixture_version": PROMPT_REGRESSION_FIXTURE_VERSION,
        "inputs_fingerprint": _compact_json(
            {
                "date": date,
                "brief_goal": brief_goal,
                "topics": [topic.name for topic in topics],
                "headline_candidate_ids": [item.id for item in candidates],
                "selected_article_ids": [item.candidate.id for item in selected_articles],
                "prior_report_ids": [item.id for item in prior_reports],
            }
        ),
        "stages": stages,
        "response_shapes": response_shape_fixtures(),
    }
