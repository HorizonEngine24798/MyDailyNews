from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from mydailynews.app.models import AppConfig, BriefOutput, PerspectivesReportConfig
from mydailynews.common.utils import canonical_article_url, stable_id
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.perspectives.sources import load_source_registry, validate_source_registry
from mydailynews.pipeline.perspectives_report import (
    PERSPECTIVES_PLANNER_SCHEMA,
    _claim_card_score,
    _normalize_framing_response,
    _normalize_claim_synthesis_response,
    _fetch_selected_article_contexts,
    _rank_coverage_rows,
    _relevant_coverage_rows,
    _retrieval_requests,
    _run_provider_requests,
    _select_sources_for_plan,
    _thread_summary,
    _validate_planner_response,
    _visible_claim_cards,
    _with_article_context,
    build_framing_comparisons,
    collect_perspectives_inputs,
    plan_perspectives_queries,
    render_perspectives_report_markdown,
    run_perspectives_report,
)
from mydailynews.pipeline.narrative_brief import run_narrative_brief


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = REPO_ROOT / ".codex_tmp_test" / "perspectives_report"


class FakeReporter:
    def phase(self, message: str) -> None:
        return None


class FakeAIConfig:
    backend = "fake"
    response_format = "json_object"

    @property
    def effective_model_label(self) -> str:
        return "fake-perspectives-model"


class FakeAIClient:
    config = FakeAIConfig()

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.max_input_tokens = 12000
        self.max_new_tokens = 1800
        self.calls: list[dict] = []

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def complete_json(self, system: str, user: str, label: str = "ai.complete_json", **kwargs) -> dict:
        self.calls.append({"system": system, "user": user, "label": label, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError(f"unexpected AI call: {label}")
        return self.responses.pop(0)

    def unload(self) -> None:
        return None


def _registry_sources() -> list[dict]:
    return [
        {
            "source_id": "gb_test",
            "name": "GB Test",
            "country": "GB",
            "language": "en",
            "source_type": "newspaper",
            "homepage_url": "https://gb.example/",
            "feed_urls": ["https://gb.example/feed"],
            "sitemap_urls": [],
            "regions": ["europe"],
            "category": "world",
            "tags": ["world", "europe", "gb"],
            "enabled": True,
        },
        {
            "source_id": "fr_test",
            "name": "FR Test",
            "country": "FR",
            "language": "fr",
            "source_type": "newspaper",
            "homepage_url": "https://fr.example/",
            "feed_urls": ["https://fr.example/feed"],
            "sitemap_urls": [],
            "regions": ["europe"],
            "category": "world",
            "tags": ["world", "europe", "fr"],
            "enabled": True,
        },
        {
            "source_id": "jp_test",
            "name": "JP Test",
            "country": "JP",
            "language": "ja",
            "source_type": "newspaper",
            "homepage_url": "https://jp.example/",
            "feed_urls": ["https://jp.example/feed"],
            "sitemap_urls": [],
            "regions": ["east_asia"],
            "category": "world",
            "tags": ["world", "east_asia", "jp"],
            "enabled": True,
        },
    ]


def _planner_response() -> dict:
    plan = {
        "story_id": "story-1",
        "queries": [
            "Shared story diplomatic summit",
            "Shared story leaders deal",
            "Shared story regional reaction",
        ],
        "target_tags": {
            "countries": ["United Kingdom", "France", "Japan"],
            "regions": ["Europe", "East Asia"],
            "languages": ["English", "French", "Japanese"],
        },
    }
    return {"plans": [plan]}


def _framing_response() -> dict:
    return {
        "stories": [
            {
                "story_id": "story-1",
                "synthesis": "The sampled articles share the same event but lead with different local stakes.",
                "shared_facts": [{"text": "All articles describe the same diplomatic story.", "article_ids": ["local-gb"]}],
                "country_source_comparison": [
                    {"text": "GB coverage foregrounds logistics while FR coverage foregrounds diplomatic language.", "article_ids": ["local-gb", "local-fr"]}
                ],
                "language_differences": [{"text": "French-language coverage uses more institutional phrasing.", "article_ids": ["local-fr"]}],
                "coverage_limitations": ["Small test fixture."],
            }
        ]
    }


def _coverage_row(provider: str, url: str, *, country: str, language: str, title: str = "Shared story live angle") -> dict:
    return {
        "article_id": f"{provider}-{country}-{language}",
        "provider": provider,
        "provider_id": f"{provider}-{country}-{language}",
        "url": url,
        "canonical_url": canonical_article_url(url),
        "domain": url.split("/")[2],
        "source_name": f"{country} {provider}",
        "source_country": country,
        "source_language": language,
        "source_key": f"{provider}|{country}|{language}",
        "title": title,
        "snippet": f"Shared story snippet from {country}",
        "published_at": "2026-07-08T12:00:00Z",
        "image_url": "",
        "context_status": "metadata_only",
        "context_text": (
            f"Shared story context from {country} via {provider}. "
            "Officials, regional analysts, and local stakeholders describe the same diplomatic summit with enough surrounding detail "
            "to support a framing comparison in the fixture. The paragraph deliberately runs long enough to be treated as a "
            "meaningful lead rather than a short feed stub, while still staying compact for test readability."
        ),
        "exact_duplicate_of": "",
    }


class FakeGdeltRetriever:
    instances: list["FakeGdeltRetriever"] = []
    rows: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[dict] = []
        self.__class__.instances.append(self)

    def search(self, query: str, **kwargs) -> tuple[list[dict], list[str]]:
        self.calls.append({"query": query, **kwargs})
        return list(self.__class__.rows), []


class FakeRegistryRssRetriever:
    instances: list["FakeRegistryRssRetriever"] = []
    rows: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[dict] = []
        self.__class__.instances.append(self)

    def search(self, query: str, **kwargs) -> tuple[list[dict], list[str]]:
        self.calls.append({"query": query, **kwargs})
        return list(self.__class__.rows), []


class FakeGNewsRetriever:
    instances: list["FakeGNewsRetriever"] = []
    rows: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        self.api_key = kwargs.get("api_key", "")
        self.calls: list[dict] = []
        self.__class__.instances.append(self)

    def search(self, query: str, **kwargs) -> tuple[list[dict], list[str]]:
        self.calls.append({"query": query, **kwargs})
        return list(self.__class__.rows), []


class FakeArticleRetriever:
    instances: list["FakeArticleRetriever"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[str] = []
        self.__class__.instances.append(self)

    def fetch_text_with_url(self, url: str) -> tuple[str, str, str]:
        self.calls.append(url)
        return (
            "Officials described the shared diplomatic summit and its immediate outcome in detail. " * 5
            + "\n\nRegional analysts explained the local consequences and the next scheduled talks. " * 5,
            "ok",
            url,
        )


class RateLimitedGdeltRetriever:
    rate_limited = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, **kwargs) -> tuple[list[dict], list[str]]:
        self.calls.append(query)
        self.rate_limited = True
        return [], ["gdelt_doc: request failed for 'q' (status 429)."]


class PerspectivesReportModuleTests(unittest.TestCase):
    def _temp_dir(self) -> Path:
        path = TEMP_ROOT / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path

    def test_claim_synthesis_validates_complete_relationships_and_trusted_sources(self) -> None:
        warnings: list[str] = []
        raw = {
            "story_id": "story-1",
            "framing_report": {
                "synthesis": "Most coverage repeats one origin.",
                "synthesis_article_ids": ["a1"],
                "shared_facts": [],
                "repetition_without_independent_support": [{"text": "The claim is repeated.", "article_ids": ["a1"]}],
                "verified_or_independently_supported_claims": [],
                "qualified_disputed_or_unresolved_claims": [],
                "country_source_comparison": [],
                "coverage_limitations": [],
            },
            "claim_perspectives": [
                {
                    "claim_id": "claim-1",
                    "importance_reason": "It sets the date.",
                    "claimants": [],
                    "coverage": [{"article_id": "a1", "relationship": "reports_or_quotes", "evidence_basis": "attribution", "explanation": "Repeats the announcement."}],
                    "not_covered_article_ids": ["a2"],
                    "synthesis": "One outlet repeats it.",
                    "coverage_limitations": [],
                    "card": {"title": "Start date", "who_says": "The ministry.", "reporting_summary": "One outlet repeats it.", "evidence_check": "Not checked.", "qualification": "", "limitations": "No primary record."},
                }
            ],
        }
        story = {
            "story_id": "story-1",
            "claims": [{"claim_id": "claim-1", "claim": "The measure starts 1 August.", "claim_type": "factual"}],
            "articles": [
                {"article_id": "a1", "title": "Report", "source_name": "Outlet", "url": "https://trusted.example/report"},
                {"article_id": "a2", "title": "Other", "source_name": "Other", "url": "javascript:alert(1)"},
            ],
        }

        result = _normalize_claim_synthesis_response(
            raw,
            story=story,
            known_article_ids=["a1", "a2"],
            verification_by_claim={},
            verification_documents={},
            warnings=warnings,
        )

        claim = result["claim_perspectives"][0]
        self.assertEqual(claim["coverage"][0]["relationship"], "reports_or_quotes")
        self.assertEqual(claim["not_covered_article_ids"], ["a2"])
        self.assertEqual(claim["card"]["sources"][0]["url"], "https://trusted.example/report")
        self.assertFalse(warnings)

    def test_visible_claim_cards_drop_routine_consensus_and_keep_evidentiary_tension(self) -> None:
        routine_card = {"claim": "Strikes continued for a ninth day.", "qualification": "", "limitations": ""}
        routine_score = _claim_card_score(
            {"claim_type": "factual"},
            [
                {"relationship": "reports_or_quotes"},
                {"relationship": "reports_or_quotes"},
                {"relationship": "context_only"},
            ],
            {"status": "not_selected"},
            routine_card,
        )
        disputed_card = {
            "claim": "The policy caused the market decline.",
            "qualification": "Other reporting points to wider market pressure.",
            "limitations": "Causation remains unresolved.",
        }
        disputed_score = _claim_card_score(
            {"claim_type": "causal"},
            [{"relationship": "disputes"}, {"relationship": "independently_supports"}],
            {"status": "checked", "verdict": "mixed"},
            disputed_card,
        )
        routine_card["editorial_score"] = routine_score
        disputed_card["editorial_score"] = disputed_score

        visible = _visible_claim_cards([{"card": routine_card}, {"card": disputed_card}])

        self.assertEqual(visible, [disputed_card])

    def test_evidence_claim_reaches_verified_card_and_valid_narrative_marker(self) -> None:
        output_dir = self._temp_dir()
        date = "2026-07-08"
        claim_text = "The next diplomatic talks are scheduled for September."
        claim_id = f"claim-{stable_id(claim_text.lower())}"
        brief_path = output_dir / f"{date}_general_brief.json"
        brief_path.write_text(
            json.dumps(
                {
                    "title": "Daily Brief",
                    "selected_articles": [
                        {
                            "id": "seed-a",
                            "headline": "Diplomatic summit sets next talks",
                            "source": "Seed News",
                            "url": "https://seed.example/summit",
                        }
                    ],
                    "analysis": {
                        "evidence_packet": {
                            "story_clusters": [
                                {
                                    "cluster_id": "story-1",
                                    "article_ids": ["seed-a", "outside"],
                                    "key_claims": [
                                        {
                                            "claim": claim_text,
                                            "claimant": "The foreign ministry",
                                            "claim_type": "factual",
                                            "support_article_ids": ["seed-a", "outside"],
                                            "origin_article_ids": ["seed-a", "outside"],
                                        }
                                    ],
                                },
                                {
                                    "cluster_id": "other-story",
                                    "article_ids": ["outside"],
                                    "key_claims": [
                                        {
                                            "claim": "An unrelated claim must not cross the story boundary.",
                                            "support_article_ids": ["outside"],
                                        }
                                    ],
                                },
                            ]
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        enrichment_path = output_dir / f"{date}_enrichment.json"
        enrichment_path.write_text(
            json.dumps(
                {
                    "source_briefs": ["general"],
                    "selected_articles": [
                        {
                            "id": "seed-a",
                            "headline": "Diplomatic summit sets next talks",
                            "source": "Seed News",
                            "url": "https://seed.example/summit",
                        }
                    ],
                    "story_threads": [
                        {
                            "story_id": "story-1",
                            "story_title": "Diplomatic summit sets next talks",
                            "summary": "Officials announced another round of diplomatic talks.",
                            "article_ids": ["seed-a"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        source_output = BriefOutput(
            name="general",
            markdown_path=str(output_dir / f"{date}_general_brief.md"),
            json_path=str(brief_path),
            candidate_count=1,
            selected_count=1,
        )

        inputs = collect_perspectives_inputs(
            output_dir=output_dir,
            date=date,
            source_outputs=[source_output],
            enrichment_json_path=str(enrichment_path),
            allow_disk_fallback=False,
        )
        self.assertEqual(inputs["stories"][0]["claims"][0]["support_article_ids"], ["seed-a"])
        self.assertEqual(inputs["stories"][0]["claims"][0]["origin_article_ids"], ["seed-a"])
        self.assertTrue(any("no supporting-article overlap" in warning for warning in inputs["warnings"]))

        verification_url = "https://verify.example/september-talks"
        verification_title = "Official record confirms September talks"
        document_id = f"verification-{stable_id(canonical_article_url(verification_url), verification_title)}"
        coverage_article = _coverage_row(
            "registry_rss",
            "https://gb.example/summit-angle",
            country="GB",
            language="en",
        )
        summary_ai = FakeAIClient(
            [
                {
                    "plans": [
                        {
                            "story_id": "story-1",
                            "queries": [
                                "diplomatic summit next talks",
                                "summit regional reaction",
                                "summit timeline",
                            ],
                            "target_tags": {
                                "countries": ["United Kingdom"],
                                "regions": ["Europe"],
                                "languages": ["English"],
                            },
                            "verification_targets": [
                                {
                                    "claim_id": claim_id,
                                    "importance_reason": "The date is material.",
                                    "required_evidence_types": ["independent"],
                                    "queries": [
                                        {
                                            "query": "September diplomatic talks official record",
                                            "evidence_type": "independent",
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                },
                {
                    "claim_id": claim_id,
                    "verdict": "supported",
                    "verdict_scope": "published_schedule",
                    "supporting_evidence": [
                        {
                            "document_id": document_id,
                            "evidence_basis": "published schedule",
                            "explanation": "The record names September.",
                        }
                    ],
                    "contradicting_evidence": [],
                    "reasoning_summary": "The independent record supports the announced month.",
                    "source_independence": "Independent of the seed report.",
                    "limitations": ["The exact day is not stated."],
                    "what_would_change_the_verdict": "A superseding official notice.",
                },
                {
                    "story_id": "story-1",
                    "framing_report": {
                        "synthesis": "Coverage agrees that another round of talks was announced.",
                        "synthesis_article_ids": [coverage_article["article_id"]],
                        "shared_facts": [
                            {
                                "text": "Another round of talks was announced.",
                                "article_ids": [coverage_article["article_id"]],
                            }
                        ],
                        "repetition_without_independent_support": [],
                        "verified_or_independently_supported_claims": [
                            {
                                "text": "An independent record supports the September schedule.",
                                "article_ids": [coverage_article["article_id"]],
                            }
                        ],
                        "qualified_disputed_or_unresolved_claims": [],
                        "country_source_comparison": [],
                        "coverage_limitations": [],
                    },
                    "claim_perspectives": [
                        {
                            "claim_id": claim_id,
                            "importance_reason": "The timing shapes the diplomatic outlook.",
                            "claimants": [
                                {
                                    "name": "The foreign ministry",
                                    "role": "originator",
                                    "evidence_basis": "announcement",
                                    "article_ids": [coverage_article["article_id"]],
                                }
                            ],
                            "coverage": [
                                {
                                    "article_id": coverage_article["article_id"],
                                    "relationship": "reports_or_quotes",
                                    "evidence_basis": "attributed report",
                                    "explanation": "The outlet reports the announcement.",
                                }
                            ],
                            "not_covered_article_ids": [],
                            "synthesis": "Reporting repeats the announcement; independent evidence supports the month.",
                            "coverage_limitations": [],
                            "card": {
                                "title": "September talks",
                                "who_says": "The foreign ministry announced the schedule.",
                                "reporting_summary": "Coverage repeats the announcement.",
                                "evidence_check": "An independent record supports September.",
                                "qualification": "The exact day remains unspecified.",
                                "limitations": "A later notice could change the schedule.",
                            },
                        }
                    ],
                },
            ]
        )
        config = AppConfig(
            output_dir=str(output_dir),
            perspectives_report=PerspectivesReportConfig(enabled=True),
        )
        orchestrator = SimpleNamespace(
            config=config,
            warnings=[],
            reporter=FakeReporter(),
            debug=DebugLogger(False),
            summary_ai_client=summary_ai,
        )
        FakeGdeltRetriever.instances = []
        FakeGdeltRetriever.rows = [
            _coverage_row(
                "gdelt_doc",
                verification_url,
                country="GB",
                language="en",
                title=verification_title,
            )
        ]
        FakeArticleRetriever.instances = []
        coverage = {
            "story-1": {
                "coverage_articles": [coverage_article],
                "coverage_quality": {"status": "ok", "thin_reasons": []},
            }
        }
        with patch("mydailynews.pipeline.perspectives_report.load_source_registry", return_value=_registry_sources()), patch(
            "mydailynews.pipeline.perspectives_report.collect_global_coverage", return_value=coverage
        ), patch("mydailynews.pipeline.perspectives_report.GdeltDocRetriever", FakeGdeltRetriever), patch(
            "mydailynews.pipeline.perspectives_report.ArticleRetriever", FakeArticleRetriever
        ):
            perspectives_output = run_perspectives_report(
                orchestrator,
                date=date,
                outputs=[source_output],
                enrichment_json_path=str(enrichment_path),
                allow_disk_fallback=False,
            )

        self.assertIsNotNone(perspectives_output)
        perspectives = json.loads(Path(perspectives_output.json_path).read_text(encoding="utf-8"))
        story = perspectives["stories"][0]
        self.assertEqual(perspectives["evidence_diagnostics"], {"status": "claims_available", "claims": 1})
        self.assertEqual(perspectives["verification_diagnostics"]["status"], "verification_completed")
        self.assertEqual(story["planner"]["verification_targets"][0]["status"], "selected")
        self.assertEqual(story["verification_documents"][0]["document_id"], document_id)
        self.assertEqual(story["claim_perspectives"][0]["verification"]["verdict"], "supported")
        self.assertEqual(perspectives["claim_card_diagnostics"], {"status": "cards_produced", "cards": 1})
        self.assertIn(document_id, {source["article_id"] for source in story["claim_context_cards"][0]["sources"]})

        narrative_ai = FakeAIClient(
            [
                {
                    "title": "Narrative Daily Brief",
                    "lede": "The next talks are scheduled for September. <<1>> Invalid marker. <<99>>",
                    "segments": [],
                    "closing": "That is the briefing.",
                }
            ]
        )
        narrative_orchestrator = SimpleNamespace(
            config=config,
            warnings=[],
            reporter=FakeReporter(),
            debug=DebugLogger(False),
            final_ai_client=narrative_ai,
            _stage_payload=lambda **kwargs: kwargs,
            _record_stage_artifact=lambda **_kwargs: None,
        )
        narrative_output = run_narrative_brief(
            narrative_orchestrator,
            outputs=[source_output],
            date=date,
            use_enrichment=False,
            perspectives_json_path=perspectives_output.json_path,
            allow_disk_fallback=False,
        )

        self.assertIsNotNone(narrative_output)
        narrative = json.loads(Path(narrative_output.json_path).read_text(encoding="utf-8"))
        self.assertIn("The exact day remains unspecified", narrative_ai.calls[0]["user"])
        self.assertEqual(narrative["claim_context_cards"][0]["claim_id"], claim_id)
        self.assertEqual(narrative["claim_context_cards"][0]["ref"], 1)
        self.assertIn("<<1>>", narrative["lede"])
        self.assertNotIn("<<99>>", narrative["lede"])

    def test_planner_drives_sources_and_primary_providers_run(self) -> None:
        output_dir = self._temp_dir()
        (output_dir / "2026-07-08_enrichment.json").write_text(
            json.dumps(
                {
                    "selected_articles": [
                        {
                            "id": "a1",
                            "headline": "Shared story diplomatic summit",
                            "source": "Seed",
                            "url": "https://seed.example/story",
                            "topic": "World",
                        }
                    ],
                    "story_threads": [
                        {
                            "story_id": "story-1",
                            "story_title": "Shared story diplomatic summit",
                            "summary": "A diplomatic summit drew different regional reactions.",
                            "entities": ["Summit"],
                            "article_ids": ["a1"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        ai = FakeAIClient([_planner_response(), _framing_response()])
        orchestrator = SimpleNamespace(
            config=AppConfig(
                output_dir=str(output_dir),
                perspectives_report=PerspectivesReportConfig(
                    enabled=True,
                    gnews_api_key="test-key",
                    coverage_max_records_per_story=8,
                    minimum_source_countries=2,
                ),
            ),
            warnings=[],
            reporter=FakeReporter(),
            debug=DebugLogger(False),
            summary_ai_client=ai,
        )

        FakeGdeltRetriever.instances = []
        FakeGdeltRetriever.rows = [_coverage_row("gdelt_doc", "https://gdelt.example/shared-story", country="DE", language="de")]
        FakeRegistryRssRetriever.instances = []
        FakeRegistryRssRetriever.rows = [_coverage_row("registry_rss", "https://rss.example/shared-story", country="AU", language="en")]
        FakeGNewsRetriever.instances = []
        FakeGNewsRetriever.rows = [_coverage_row("gnews", "https://gnews.example/shared-story", country="CA", language="en")]
        FakeArticleRetriever.instances = []
        with patch("mydailynews.pipeline.perspectives_report.GdeltDocRetriever", FakeGdeltRetriever), patch(
            "mydailynews.pipeline.perspectives_report.RegistryRssRetriever",
            FakeRegistryRssRetriever,
        ), patch(
            "mydailynews.pipeline.perspectives_report.GNewsRetriever",
            FakeGNewsRetriever,
        ), patch(
            "mydailynews.pipeline.perspectives_report.ArticleRetriever",
            FakeArticleRetriever,
        ), patch(
            "mydailynews.pipeline.perspectives_report.load_source_registry",
            return_value=_registry_sources(),
        ):
            output = run_perspectives_report(orchestrator, date="2026-07-08")

        payload = json.loads(Path(output.json_path).read_text(encoding="utf-8"))
        story = payload["stories"][0]
        self.assertEqual([call["label"] for call in ai.calls], ["perspectives planner", "perspectives framing report"])
        self.assertIs(ai.calls[0]["kwargs"]["json_schema"], PERSPECTIVES_PLANNER_SCHEMA)
        self.assertIn("tag_options", ai.calls[0]["user"])
        self.assertIn("Do not return source_id values", ai.calls[0]["user"])
        self.assertEqual(story["planner"]["target_tags"]["normalized"]["countries"], ["GB", "FR", "JP"])
        self.assertEqual({source["source_id"] for source in story["selected_sources"]}, {"gb_test", "fr_test", "jp_test"})
        self.assertIn("registry_rss", story["provider_statuses"])
        self.assertIn("gnews", story["provider_statuses"])
        self.assertGreater(len(FakeRegistryRssRetriever.instances[0].calls), 0)
        self.assertGreater(len(FakeGNewsRetriever.instances[0].calls), 0)
        self.assertGreater(len(FakeGdeltRetriever.instances[0].calls), 0)
        self.assertLessEqual(story["coverage_counts"]["articles"], 8)
        self.assertIn("fetched_article", story["coverage_counts"]["context_statuses"])
        self.assertGreater(story["funnel_counts"]["overall"]["raw"], 0)
        self.assertGreater(len(FakeArticleRetriever.instances[0].calls), 0)
        self.assertEqual(story["framing_report"]["synthesis"], _framing_response()["stories"][0]["synthesis"])
        self.assertIn("Never put article IDs in prose fields", ai.calls[-1]["user"])

    def test_rss_fallback_runs_when_global_providers_return_no_rows(self) -> None:
        output_dir = self._temp_dir()
        (output_dir / "2026-07-08_enrichment.json").write_text(
            json.dumps(
                {
                    "selected_articles": [{"id": "a1", "headline": "Shared story diplomatic summit", "source": "Seed", "url": "https://seed.example/story"}],
                    "story_threads": [{"story_id": "story-1", "story_title": "Shared story diplomatic summit", "article_ids": ["a1"]}],
                }
            ),
            encoding="utf-8",
        )
        ai = FakeAIClient([_planner_response(), _framing_response()])
        orchestrator = SimpleNamespace(
            config=AppConfig(
                output_dir=str(output_dir),
                perspectives_report=PerspectivesReportConfig(enabled=True, gnews_api_key="test-key"),
            ),
            warnings=[],
            reporter=FakeReporter(),
            debug=DebugLogger(False),
            summary_ai_client=ai,
        )

        FakeGdeltRetriever.instances = []
        FakeGdeltRetriever.rows = []
        FakeRegistryRssRetriever.instances = []
        FakeRegistryRssRetriever.rows = [_coverage_row("registry_rss", "https://rss.example/shared-story", country="AU", language="en")]
        FakeGNewsRetriever.instances = []
        FakeGNewsRetriever.rows = []
        FakeArticleRetriever.instances = []
        with patch("mydailynews.pipeline.perspectives_report.GdeltDocRetriever", FakeGdeltRetriever), patch(
            "mydailynews.pipeline.perspectives_report.RegistryRssRetriever",
            FakeRegistryRssRetriever,
        ), patch(
            "mydailynews.pipeline.perspectives_report.GNewsRetriever",
            FakeGNewsRetriever,
        ), patch(
            "mydailynews.pipeline.perspectives_report.ArticleRetriever",
            FakeArticleRetriever,
        ), patch(
            "mydailynews.pipeline.perspectives_report.load_source_registry",
            return_value=_registry_sources(),
        ):
            output = run_perspectives_report(orchestrator, date="2026-07-08")

        story = json.loads(Path(output.json_path).read_text(encoding="utf-8"))["stories"][0]
        self.assertIn("registry_rss", story["provider_statuses"])
        self.assertGreater(len(FakeRegistryRssRetriever.instances[0].calls), 0)

    def test_planner_plans_recovers_valid_items_and_normalizes_aliases(self) -> None:
        stories = [
            {"story_id": "story-1", "story_title": "Story one", "article_ids": ["a1"]},
            {"story_id": "story-2", "story_title": "Story two", "article_ids": ["a2"]},
        ]
        valid = {
            "story_id": "story-1",
            "queries": ["Story one event", "Story one reaction", "Story one timeline"],
            "target_tags": {"countries": ["U.S.", "UK"], "regions": ["Middle East"], "languages": ["English"]},
        }
        warnings: list[str] = []

        plans = _validate_planner_response(
            {"plans": ["not an object", valid]},
            stories=stories,
            source_registry=_registry_sources(),
            warnings=warnings,
        )

        self.assertEqual(plans["story-1"]["target_tags"]["normalized"]["countries"], ["US", "GB"])
        self.assertEqual(plans["story-1"]["target_tags"]["normalized"]["languages"], ["en"])
        self.assertEqual(plans["story-1"]["status"], "ok")
        self.assertEqual(plans["story-2"]["status"], "planner_failed")
        self.assertTrue(any("planner plan 1 was not an object" in warning for warning in warnings))

    def test_planner_accepts_structured_plans_value(self) -> None:
        stories = [{"story_id": "story-1", "story_title": "Story one", "article_ids": ["a1"]}]
        raw_plan = {
            "story_id": "story-1",
            "queries": ["Story one event", "Story one reaction", "Story one timeline"],
            "target_tags": {"countries": ["UK"], "regions": ["Europe"], "languages": ["English"]},
        }
        warnings: list[str] = []

        plans = _validate_planner_response({"plans": [raw_plan]}, stories=stories, source_registry=_registry_sources(), warnings=warnings)

        self.assertEqual(plans["story-1"]["status"], "ok")
        self.assertFalse(any("malformed JSON" in warning for warning in warnings))

    def test_planner_schema_requires_each_tag_group(self) -> None:
        tag_properties = PERSPECTIVES_PLANNER_SCHEMA.schema["properties"]["plans"]["items"]["properties"]["target_tags"]["properties"]

        self.assertEqual(tag_properties["countries"]["minItems"], 1)
        self.assertEqual(tag_properties["regions"]["minItems"], 1)
        self.assertEqual(
            PERSPECTIVES_PLANNER_SCHEMA.schema["properties"]["plans"]["items"]["properties"]["target_tags"]["required"],
            ["countries", "regions"],
        )

    def test_anchor_groups_add_bounded_anchor_requests(self) -> None:
        requests = _retrieval_requests(
            {
                "queries": ["Acme supply deal latest"],
                "anchor_groups": [
                    {"kind": "entity", "terms": ["Acme Corporation"]},
                    {"kind": "event", "terms": ["supply agreement", "factory investment"]},
                ],
            },
            _registry_sources(),
        )

        self.assertEqual(requests[0]["query_type"], "canonical")
        anchor_requests = [request for request in requests if request["query_type"] == "anchor"]
        self.assertLessEqual(len(anchor_requests), 3)
        self.assertTrue(anchor_requests)
        self.assertIn("Acme Corporation", anchor_requests[0]["query"])
        self.assertTrue(all("source_languages" not in request for request in requests))

        selected, _ = _select_sources_for_plan(
            {
                "target_tags": {
                    "normalized": {
                        "countries": ["GB"],
                        "regions": [],
                        "languages": ["ja"],
                    }
                }
            },
            _registry_sources(),
            SimpleNamespace(),
        )
        self.assertEqual([source["source_id"] for source in selected], ["gb_test"])

    def test_canonical_queries_drop_zero_overlap_seed_contamination(self) -> None:
        summary = _thread_summary(
            {
                "internal_articles": [
                    {
                        "summary": (
                            "President Trump said the administration pursued a ceasefire after a retaliatory "
                            "explosion near Hormuz."
                        )
                    }
                ]
            }
        )
        requests = _retrieval_requests(
            {
                "queries": [
                    "Trump administration voter data election security",
                    "ceasefire retaliatory explosion",
                    "Iran Hormuz strikes red line",
                ],
                "anchor_groups": [],
            },
            _registry_sources(),
            story={"story_title": "Escalating US-Iran Conflict and Strait of Hormuz Tensions", "summary": summary},
        )

        self.assertEqual(
            [request["query"] for request in requests],
            ["ceasefire retaliatory explosion", "Iran Hormuz strikes red line"],
        )

    def test_verification_targets_do_not_replace_canonical_queries(self) -> None:
        canonical = ["Iran Hormuz strikes", "regional shipping response", "ceasefire diplomacy"]
        stories = [
            {
                "story_id": "story-1",
                "story_title": "Iran and the Strait of Hormuz",
                "article_ids": ["a1"],
                "claims": [{"claim_id": "claim-1", "claim": "Shipping was disrupted."}],
            }
        ]
        raw_plan = {
            "story_id": "story-1",
            "queries": canonical,
            "target_tags": {"countries": ["GB"], "regions": ["europe"]},
            "verification_targets": [
                {
                    "claim_id": "claim-1",
                    "importance_reason": "Material shipping claim",
                    "required_evidence_types": ["independent"],
                    "queries": [{"query": "independent Hormuz shipping data", "evidence_type": "independent"}],
                }
            ],
        }

        plans = _validate_planner_response(
            {"plans": [raw_plan]},
            stories=stories,
            source_registry=_registry_sources(),
            warnings=[],
        )

        self.assertEqual(plans["story-1"]["queries"], canonical)
        self.assertEqual(
            plans["story-1"]["verification_targets"][0]["queries"],
            [{"query": "independent Hormuz shipping data", "evidence_type": "independent"}],
        )

    def test_planner_retries_query_only_plan(self) -> None:
        ai = FakeAIClient(
            [
                {
                    "plans": [
                        {
                            "story_id": "story-1",
                            "queries": ["Story one event", "Story one reaction", "Story one timeline"],
                            "target_tags": {"countries": [], "regions": [], "languages": []},
                        }
                    ]
                },
                _planner_response(),
            ]
        )
        warnings: list[str] = []

        plans = plan_perspectives_queries(
            SimpleNamespace(summary_ai_client=ai),
            date="2026-07-08",
            inputs={
                "articles": [{"id": "a1", "headline": "Shared story diplomatic summit", "source": "Seed"}],
                "stories": [{"story_id": "story-1", "story_title": "Shared story diplomatic summit", "article_ids": ["a1"]}],
            },
            config=PerspectivesReportConfig(enabled=True),
            source_registry=_registry_sources(),
            warnings=warnings,
        )

        self.assertEqual(plans["story-1"]["status"], "ok")
        self.assertEqual([call["label"] for call in ai.calls], ["perspectives planner", "perspectives planner"])
        self.assertIn("Retry instruction", ai.calls[1]["user"])
        self.assertFalse(warnings)

    def test_planner_splits_large_runs_into_bounded_calls(self) -> None:
        stories = [
            {"story_id": f"story-{number}", "story_title": f"Story {number}", "article_ids": []}
            for number in range(1, 6)
        ]

        def plan(story_id: str) -> dict:
            return {
                "story_id": story_id,
                "queries": [f"{story_id} event", f"{story_id} reaction", f"{story_id} timeline"],
                "target_tags": {"countries": ["GB"], "regions": ["europe"]},
                "verification_targets": [],
            }

        ai = FakeAIClient(
            [
                {"plans": [plan(story["story_id"]) for story in stories[:4]]},
                {"plans": [plan(stories[4]["story_id"])]},
            ]
        )

        plans = plan_perspectives_queries(
            SimpleNamespace(summary_ai_client=ai),
            date="2026-07-08",
            inputs={"articles": [], "stories": stories},
            config=PerspectivesReportConfig(enabled=True),
            source_registry=_registry_sources(),
            warnings=[],
        )

        self.assertTrue(all(plan["status"] == "ok" for plan in plans.values()))
        self.assertEqual(
            [call["label"] for call in ai.calls],
            ["perspectives planner batch 1/2", "perspectives planner batch 2/2"],
        )
        self.assertNotIn('"story_id":"story-5"', ai.calls[0]["user"])
        self.assertIn('"story_id":"story-5"', ai.calls[1]["user"])

    def test_framing_runs_once_per_story(self) -> None:
        ai = FakeAIClient(
            [
                {
                    "stories": [
                        {
                            "story_id": "story-1",
                            "synthesis": "Story one framing.",
                            "shared_facts": [{"text": "Shared event one.", "article_ids": ["registry_rss-GB-en"]}],
                            "country_source_comparison": [],
                            "language_differences": [],
                            "coverage_limitations": [],
                        }
                    ]
                },
                {
                    "stories": [
                        {
                            "story_id": "story-2",
                            "synthesis": "Story two framing.",
                            "shared_facts": [{"text": "Shared event two.", "article_ids": ["registry_rss-FR-fr"]}],
                            "country_source_comparison": [],
                            "language_differences": [],
                            "coverage_limitations": [],
                        }
                    ]
                },
            ]
        )
        warnings: list[str] = []

        reports = build_framing_comparisons(
            SimpleNamespace(summary_ai_client=ai),
            inputs={
                "stories": [
                    {"story_id": "story-1", "story_title": "Shared story one", "summary": "First story."},
                    {"story_id": "story-2", "story_title": "Shared story two", "summary": "Second story."},
                ]
            },
            coverage_by_story={
                "story-1": {
                    "coverage_articles": [_coverage_row("registry_rss", "https://gb.example/shared-one", country="GB", language="en")],
                    "coverage_quality": {"thin_reasons": []},
                },
                "story-2": {
                    "coverage_articles": [_coverage_row("registry_rss", "https://fr.example/shared-two", country="FR", language="fr")],
                    "coverage_quality": {"thin_reasons": []},
                },
            },
            warnings=warnings,
        )

        self.assertEqual([call["label"] for call in ai.calls], ["perspectives framing report", "perspectives framing report"])
        self.assertIn('"story_id":"story-1"', ai.calls[0]["user"])
        self.assertNotIn('"story_id":"story-2"', ai.calls[0]["user"])
        self.assertIn('"story_id":"story-2"', ai.calls[1]["user"])
        self.assertEqual(ai.calls[0]["kwargs"]["max_new_tokens"], ai.max_new_tokens)
        self.assertEqual(reports["story-1"]["synthesis"], "Story one framing.")
        self.assertEqual(reports["story-2"]["synthesis"], "Story two framing.")
        self.assertFalse(warnings)

    def test_markdown_keeps_references_at_the_end(self) -> None:
        markdown = render_perspectives_report_markdown(
            {
                "date": "2026-07-08",
                "metadata": {"story_count": 1, "coverage_article_count": 1, "coverage_source_country_count": 1},
                "stories": [
                    {
                        "story_title": "Shared diplomatic story",
                        "planner": {"status": "ok"},
                        "coverage_quality": {"status": "ok", "thin_reasons": []},
                        "coverage_provider_counts": {"gdelt_doc": 1},
                        "framing_report": {
                            "synthesis": "The accounts agree on the event but emphasize different local stakes.",
                            "synthesis_article_ids": ["article-1"],
                            "shared_facts": [{"text": "The event occurred.", "article_ids": ["article-1"]}],
                            "country_source_comparison": [],
                            "coverage_limitations": [],
                        },
                        "coverage_articles": [
                            {
                                "article_id": "article-1",
                                "title": "A source headline",
                                "source_name": "Test Source",
                                "source_country": "GB",
                                "source_language": "en",
                                "canonical_url": "https://example.com/story",
                            }
                        ],
                    }
                ],
                "warnings": [],
            }
        )

        self.assertIn("### Bottom line", markdown)
        self.assertIn("## References", markdown)
        self.assertIn("- A source headline (Test Source)\n  https://example.com/story", markdown)
        self.assertNotIn("[1]", markdown)
        self.assertNotIn("`article-1`", markdown)
        self.assertGreater(markdown.index("## References"), markdown.index("### Bottom line"))

    def test_framing_output_and_markdown_do_not_cap_model_text_or_lists(self) -> None:
        article_ids = [f"article-{index}" for index in range(12)]
        synthesis = "Full synthesis " + ("x" * 1800)
        shared_facts = [
            {
                "text": f"Shared fact {index} " + ("y" * 500),
                "article_ids": article_ids,
            }
            for index in range(12)
        ]
        limitations = [f"Limitation {index} " + ("z" * 400) for index in range(10)]
        warnings: list[str] = []

        normalized = _normalize_framing_response(
            {
                "stories": [
                    {
                        "story_id": "story-1",
                        "synthesis": synthesis,
                        "synthesis_article_ids": article_ids,
                        "shared_facts": shared_facts,
                        "country_source_comparison": shared_facts,
                        "coverage_limitations": limitations,
                    }
                ]
            },
            known_story_ids=["story-1"],
            known_article_ids={"story-1": article_ids},
            warnings=warnings,
        )["story-1"]

        self.assertEqual(normalized["synthesis"], synthesis)
        self.assertEqual(normalized["synthesis_article_ids"], article_ids)
        self.assertEqual(len(normalized["shared_facts"]), 12)
        self.assertEqual(normalized["shared_facts"][-1]["text"], shared_facts[-1]["text"])
        self.assertEqual(normalized["shared_facts"][0]["article_ids"], article_ids)
        self.assertEqual(normalized["coverage_limitations"], limitations)
        self.assertFalse(warnings)

        report_warnings = [f"Warning {index}" for index in range(14)]
        markdown = render_perspectives_report_markdown(
            {
                "date": "2026-07-08",
                "metadata": {},
                "stories": [
                    {
                        "story_title": "Uncapped story",
                        "planner": {},
                        "coverage_quality": {"thin_reasons": []},
                        "framing_report": normalized,
                        "coverage_articles": [],
                    }
                ],
                "warnings": report_warnings,
            }
        )
        self.assertIn(synthesis, markdown)
        self.assertIn(shared_facts[-1]["text"], markdown)
        self.assertIn(limitations[-1], markdown)
        self.assertIn(report_warnings[-1], markdown)

    def test_coverage_relevance_filter_ignores_weak_overlap(self) -> None:
        story = {
            "story_id": "story-1",
            "story_title": "US Supreme Court Ruling on Tariffs",
            "summary": "Trump emergency tariffs face a legal challenge over trade policy.",
        }
        unrelated = _coverage_row(
            "registry_rss",
            "https://kr.example/supreme-court",
            country="KR",
            language="en",
            title="Supreme Court to rule on ex-first lady's corruption charges",
        )
        unrelated["retrieval_query"] = "US Supreme Court tariff ruling"
        unrelated["snippet"] = "Judges scheduled a verdict in a domestic corruption case."
        relevant = _coverage_row(
            "registry_rss",
            "https://us.example/tariffs",
            country="US",
            language="en",
            title="Supreme Court weighs Trump emergency tariffs",
        )
        relevant["retrieval_query"] = "Supreme Court Trump emergency tariffs illegal"
        relevant["snippet"] = "The legal challenge could reshape US trade policy and tariff authority."

        rows = _relevant_coverage_rows(_rank_coverage_rows([unrelated, relevant], story), story)

        self.assertEqual([row["title"] for row in rows], [relevant["title"]])

    def test_planner_rejects_query_only_plan(self) -> None:
        stories = [{"story_id": "story-1", "story_title": "Story one", "article_ids": ["a1"]}]
        raw_plan = {
            "story_id": "story-1",
            "queries": ["Story one event", "Story one reaction", "Story one timeline"],
            "target_tags": {"countries": [], "regions": [], "languages": []},
        }
        warnings: list[str] = []

        plans = _validate_planner_response({"plans": [raw_plan]}, stories=stories, source_registry=_registry_sources(), warnings=warnings)

        self.assertEqual(plans["story-1"]["status"], "planner_failed")
        self.assertIn("missing required target tags: countries, regions", plans["story-1"]["diagnostics"])
        self.assertIn("no usable target tags", plans["story-1"]["diagnostics"])
        self.assertTrue(any("was incomplete" in warning for warning in warnings))

    def test_source_selection_does_not_fall_back_to_active_registry(self) -> None:
        plan = {"target_tags": {"normalized": {"countries": [], "regions": [], "languages": []}}}

        selected, diagnostics = _select_sources_for_plan(plan, _registry_sources(), SimpleNamespace())

        self.assertEqual(selected, [])
        self.assertEqual(diagnostics, [])

    def test_gdelt_provider_stops_after_rate_limit(self) -> None:
        retriever = RateLimitedGdeltRetriever()
        requests = [{"query": "one"}, {"query": "two"}]

        result = _run_provider_requests(
            ("gdelt_doc", retriever),
            requests=requests,
            config=SimpleNamespace(coverage_max_records_per_story=8, coverage_timespan_days=7),
            source_by_id={},
            source_domains={},
        )

        self.assertEqual(retriever.calls, ["one"])
        self.assertEqual(result["status"], "warning")

    def test_article_context_prefers_fetched_text_then_fallbacks(self) -> None:
        long_paragraph = "This paragraph has enough substance for framing context. " * 8
        article = _with_article_context({"body": "Fetched lead.\n\n" + long_paragraph})
        self.assertEqual(article["context_status"], "fetched_article")
        self.assertIn("Fetched lead", article["context_text"])
        self.assertIn("This paragraph", article["context_text"])

        fallback = _with_article_context({"snippet": "Short feed stub."})
        self.assertEqual(fallback["context_status"], "provider_snippet")

        missing = _with_article_context({"title": "Only metadata"})
        self.assertEqual(missing["context_status"], "unavailable")
        self.assertEqual(missing["context_source"], "title_only")

    def test_article_context_uses_cache_before_page_fetch(self) -> None:
        cached_text = "Cached summit reporting with detailed evidence and attribution. " * 8
        cache = SimpleNamespace(
            get_by_aliases=lambda aliases: {
                "article_text": cached_text,
                "extraction_status": "ok",
                "resolved_url": "https://example.com/story",
            },
            store=lambda **kwargs: self.fail("cached article should not be stored again"),
        )
        retriever = SimpleNamespace(fetch_text_with_url=lambda url: self.fail("cached article should not be fetched"))

        articles = _fetch_selected_article_contexts(
            [_coverage_row("registry_rss", "https://example.com/story", country="GB", language="en")],
            story={"story_title": "Shared diplomatic summit"},
            article_retriever=retriever,
            article_text_cache=cache,
            max_workers=1,
            warnings=[],
        )

        self.assertEqual(articles[0]["context_status"], "fetched_article")
        self.assertEqual(articles[0]["context_source"], "article_cache")
        self.assertGreater(articles[0]["context_length_chars"], 0)

    def test_source_registry_validation_and_dead_feeds_inactive(self) -> None:
        sources = load_source_registry()
        self.assertGreaterEqual(len(sources), 20)
        disabled = {source["source_id"] for source in sources if not source.get("enabled", True)}
        self.assertGreaterEqual(
            disabled,
            {
                "us_ap_top",
                "jp_nhk_world",
                "il_times_of_israel",
                "tr_anadolu_en",
                "it_ansa_en",
                "bd_dhaka_tribune",
                "id_jakarta_post",
                "eu_external_action",
            },
        )
        duplicate = [sources[0], dict(sources[0])]
        self.assertTrue(any("duplicate source_id" in error for error in validate_source_registry(duplicate)))


if __name__ == "__main__":
    unittest.main()
