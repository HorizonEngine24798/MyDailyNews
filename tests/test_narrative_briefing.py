from __future__ import annotations

from pathlib import Path
import json
import unittest
import uuid

from mydailynews.ai.base import AIJsonError, JSONSchemaSpec
from mydailynews.app.models import AppConfig, BriefOutput, UserMemory
from mydailynews.briefing.narrative import (
    NarrativeBriefGenerator,
    NarrativeSourceBrief,
    render_narrative_markdown,
    strip_source_links,
    write_narrative_outputs,
)
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.pipeline.narrative_brief import run_narrative_brief


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = REPO_ROOT / ".codex_tmp_test" / "narrative_briefing"


class FakeAIClient:
    max_input_tokens = 12000
    max_new_tokens = 2048

    def __init__(self) -> None:
        self.system = ""
        self.user = ""
        self.label = ""
        self.json_schema: JSONSchemaSpec | None = None
        self.unload_count = 0

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def complete_json(
        self,
        system: str,
        user: str,
        label: str = "ai.complete_json",
        *,
        max_new_tokens: int | None = None,
        input_token_limit: int | None = None,
        json_schema: JSONSchemaSpec | None = None,
    ) -> dict:
        self.system = system
        self.user = user
        self.label = label
        self.json_schema = json_schema
        return {
            "title": "Narrative Daily Brief - 2026-06-14",
            "lede": "Good morning. The day is shaped by a policy story that needs context.",
            "segments": [
                {
                    "heading": "Policy Shift",
                    "body": "The lead story is fully explained in readable narrative form.",
                    "key_points": ["The first point is skimmable."],
                    "what_to_watch": ["The next official announcement."],
                }
            ],
            "closing": "That is the briefing.",
        }

    def unload(self) -> None:
        self.unload_count += 1


class FakeFailingAIClient(FakeAIClient):
    def complete_json(
        self,
        system: str,
        user: str,
        label: str = "ai.complete_json",
        *,
        max_new_tokens: int | None = None,
        input_token_limit: int | None = None,
        json_schema: JSONSchemaSpec | None = None,
    ) -> dict:
        self.system = system
        self.user = user
        self.label = label
        self.json_schema = json_schema
        raise RuntimeError("narrative model unavailable")


class FakeMarkdownAIClient(FakeAIClient):
    def complete_json(
        self,
        system: str,
        user: str,
        label: str = "ai.complete_json",
        *,
        max_new_tokens: int | None = None,
        input_token_limit: int | None = None,
        json_schema: JSONSchemaSpec | None = None,
    ) -> dict:
        self.system = system
        self.user = user
        self.label = label
        self.json_schema = json_schema
        raise AIJsonError(
            "invalid json",
            raw_response="# Daily Brief\n\nA readable Markdown briefing.",
        )


class FakeReporter:
    def phase(self, message: str) -> None:
        return None


class FakeOrchestrator:
    def __init__(self, output_dir: Path, client: FakeAIClient) -> None:
        self.config = AppConfig(output_dir=str(output_dir))
        self.final_ai_client = client
        self.warnings: list[str] = []
        self.reporter = FakeReporter()
        self.debug = DebugLogger(False)
        self.artifacts: list[dict] = []

    def _stage_payload(self, *, stage: str, brief_name: str, summary: dict, next_stage_input: dict) -> dict:
        return {"stage": stage, "brief_name": brief_name, "summary": summary, "next_stage_input": next_stage_input}

    def _record_stage_artifact(self, *, stage: str, brief_name: str, payload: dict) -> None:
        self.artifacts.append({"stage": stage, "brief_name": brief_name, "payload": payload})


class NarrativeBriefingTests(unittest.TestCase):
    def test_strip_source_links_removes_recursive_url_fields_and_url_text(self) -> None:
        payload = {
            "title": "Daily Brief",
            "references": [
                {
                    "title": "Example story",
                    "source": "Example News",
                    "url": "https://example.test/story",
                }
            ],
            "selected_articles": [
                {
                    "headline": "Example headline",
                    "source": "Example News",
                    "resolved_url": "https://example.test/resolved",
                    "snippet": "Read more at https://example.test/read-more for context.",
                }
            ],
            "metadata": {
                "date": "2026-06-14",
                "selection_reason_codes": {"selected": {"score_cutoff": 2}},
                "json_path": "output/2026-06-14_general_brief.json",
            },
        }

        stripped = strip_source_links(payload)

        rendered = str(stripped)
        self.assertNotIn("https://", rendered)
        self.assertNotIn("resolved_url", rendered)
        self.assertNotIn("json_path", rendered)
        self.assertNotIn("selection_reason_codes", rendered)
        self.assertIn("Example News", rendered)

    def test_generator_passes_sanitized_general_and_detailed_briefs(self) -> None:
        client = FakeAIClient()
        generator = NarrativeBriefGenerator(client, target_words=1200)
        source_briefs = [
            NarrativeSourceBrief(
                name="general",
                json_path="general.json",
                brief={
                    "title": "Daily Brief",
                    "lead": "A policy story matters.",
                    "references": [{"title": "Story", "source": "Ars Technica", "url": "https://example.test/a"}],
                },
            ),
            NarrativeSourceBrief(
                name="detailed",
                json_path="detailed.json",
                brief={
                    "title": "Detailed Brief",
                    "lead": "The detailed context changes the framing.",
                    "selected_articles": [{"headline": "Deep story", "url": "https://example.test/b"}],
                },
            ),
        ]

        result = generator.generate(source_briefs, UserMemory(role="Operator"), date="2026-06-14")

        self.assertEqual(client.label, "narrative brief generation")
        self.assertEqual(client.json_schema.name, "narrative_brief")
        self.assertNotIn("https://", client.user)
        self.assertNotIn("[pause]", client.user)
        self.assertNotIn("SSML", result["lede"])
        self.assertIn('"name":"general"', client.user)
        self.assertIn('"name":"detailed"', client.user)
        self.assertEqual(result["metadata"]["source_briefs"], ["general", "detailed"])
        self.assertIn("readable narrative", result["segments"][0]["body"])
        self.assertFalse(result["metadata"]["enrichment_used"])

    def test_generator_includes_sanitized_enrichment_when_available(self) -> None:
        client = FakeAIClient()
        generator = NarrativeBriefGenerator(client, target_words=1200)
        source_briefs = [
            NarrativeSourceBrief(
                name="general",
                json_path="general.json",
                brief={"title": "Daily Brief", "lead": "A policy story matters."},
            )
        ]
        enrichment_payload = {
            "schema_version": "enrichment_output.v1",
            "date": "2026-06-14",
            "source_briefs": ["general"],
            "selected_articles": [
                {
                    "article_text": "This full article body should not be passed into the narrative prompt.",
                    "candidate": {"url": "https://example.test/private"},
                }
            ],
            "story_threads": [
                {
                    "story_id": "story-001",
                    "story_title": "Policy context",
                    "article_ids": ["a"],
                    "status": "enriched",
                    "internal_articles": [
                        {
                            "title": "Internal context",
                            "summary": "A compact explanation that helps the narrative.",
                            "what_it_adds": "Adds useful background.",
                            "confidence": "medium",
                        }
                    ],
                }
            ],
        }

        result = generator.generate(
            source_briefs,
            UserMemory(role="Operator"),
            date="2026-06-14",
            enrichment_payload=enrichment_payload,
            enrichment_json_path="output/2026-06-14_enrichment.json",
        )

        self.assertIn("Internal context", client.user)
        self.assertNotIn("https://", client.user)
        self.assertNotIn("full article body", client.user)
        self.assertTrue(result["metadata"]["enrichment_used"])
        self.assertEqual(result["metadata"]["enrichment_json_path"], "output/2026-06-14_enrichment.json")

    def test_generator_accepts_raw_markdown_for_final_narrative(self) -> None:
        client = FakeMarkdownAIClient()
        generator = NarrativeBriefGenerator(client)
        source_briefs = [
            NarrativeSourceBrief(
                name="general",
                json_path="general.json",
                brief={"title": "Daily Brief", "lead": "A policy story matters."},
            )
        ]

        result = generator.generate(source_briefs, UserMemory(role="Operator"), date="2026-06-14")

        self.assertEqual(result["title"], "Daily Brief")
        self.assertIn("# Daily Brief", render_narrative_markdown(result))
        self.assertEqual(result["metadata"]["source_briefs"], ["general"])
        self.assertTrue(generator.warnings)

    def test_write_narrative_outputs_writes_pretty_markdown_and_json(self) -> None:
        output_dir = TEMP_ROOT / uuid.uuid4().hex
        narrative = {
            "title": "Narrative Daily Brief - 2026-06-14",
            "lede": "Good morning.\n\nHere is the full story.",
            "segments": [
                {
                    "heading": "Policy",
                    "body": "This is a polished section.",
                    "key_points": ["One useful point."],
                    "what_to_watch": ["One watch signal."],
                }
            ],
            "closing": "That is the briefing.",
            "metadata": {
                "generated_at": "2026-06-14T00:00:00+00:00",
                "source_briefs": ["general", "detailed"],
            },
        }

        markdown_path, json_path = write_narrative_outputs(output_dir, "2026-06-14", narrative)

        markdown = markdown_path.read_text(encoding="utf-8")
        self.assertTrue(markdown_path.exists())
        self.assertTrue(json_path.exists())
        self.assertIn("# Narrative Daily Brief - 2026-06-14", markdown)
        self.assertIn("## Policy", markdown)
        self.assertIn("Key points:", markdown)
        self.assertIn("What to watch:", markdown)
        self.assertIn("Source briefs: general, detailed", markdown)

    def test_pipeline_narrative_failure_returns_warning_without_raising(self) -> None:
        output_dir = TEMP_ROOT / uuid.uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=True)
        source_path = output_dir / "2026-06-14_general_brief.json"
        source_path.write_text('{"title":"Daily Brief","lead":"A policy story matters."}', encoding="utf-8")
        client = FakeFailingAIClient()
        orchestrator = FakeOrchestrator(output_dir, client)

        result = run_narrative_brief(
            orchestrator,
            outputs=[
                BriefOutput(
                    name="general",
                    markdown_path=str(output_dir / "2026-06-14_general_brief.md"),
                    json_path=str(source_path),
                    candidate_count=1,
                    selected_count=1,
                )
            ],
            date="2026-06-14",
        )

        self.assertIsNone(result)
        self.assertEqual(client.unload_count, 1)
        self.assertTrue(any("continuing with already written structured briefs" in warning for warning in orchestrator.warnings))
        self.assertFalse((output_dir / "2026-06-14_narrative_brief.md").exists())

    def test_pipeline_narrative_uses_enrichment_json_when_present(self) -> None:
        output_dir = TEMP_ROOT / uuid.uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=True)
        source_path = output_dir / "2026-06-14_general_brief.json"
        source_path.write_text('{"title":"Daily Brief","lead":"A policy story matters."}', encoding="utf-8")
        enrichment_path = output_dir / "2026-06-14_enrichment.json"
        enrichment_path.write_text(
            json.dumps(
                {
                    "schema_version": "enrichment_output.v1",
                    "date": "2026-06-14",
                    "source_briefs": ["general"],
                    "story_threads": [
                        {
                            "story_id": "story-001",
                            "story_title": "Policy context",
                            "status": "enriched",
                            "internal_articles": [
                                {"title": "Internal context", "summary": "Useful background."}
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        client = FakeAIClient()
        orchestrator = FakeOrchestrator(output_dir, client)

        result = run_narrative_brief(
            orchestrator,
            outputs=[],
            date="2026-06-14",
        )

        self.assertIsNotNone(result)
        self.assertIn("Internal context", client.user)
        written = json.loads((output_dir / "2026-06-14_narrative_brief.json").read_text(encoding="utf-8"))
        self.assertTrue(written["metadata"]["enrichment_used"])
        self.assertEqual(written["metadata"]["enrichment_json_path"], str(enrichment_path))

    def test_pipeline_narrative_ignores_disk_fallbacks_in_series_mode(self) -> None:
        output_dir = TEMP_ROOT / uuid.uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=True)
        current_general = output_dir / "current_general_brief.json"
        current_general.write_text(
            '{"title":"Current General","lead":"Fresh current-run source brief."}',
            encoding="utf-8",
        )
        (output_dir / "2026-06-14_detailed_brief.json").write_text(
            '{"title":"Stale Detailed","lead":"Stale same-day detailed source."}',
            encoding="utf-8",
        )
        (output_dir / "2026-06-14_enrichment.json").write_text(
            json.dumps(
                {
                    "schema_version": "enrichment_output.v1",
                    "date": "2026-06-14",
                    "story_threads": [
                        {
                            "story_title": "Stale enrichment context",
                            "internal_articles": [
                                {"title": "Internal context", "summary": "This should not be used."}
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        client = FakeAIClient()
        orchestrator = FakeOrchestrator(output_dir, client)

        result = run_narrative_brief(
            orchestrator,
            outputs=[
                BriefOutput(
                    name="general",
                    markdown_path=str(output_dir / "current_general_brief.md"),
                    json_path=str(current_general),
                    candidate_count=1,
                    selected_count=1,
                )
            ],
            date="2026-06-14",
            allow_disk_fallback=False,
        )

        self.assertIsNotNone(result)
        self.assertIn("Fresh current-run source brief", client.user)
        self.assertNotIn("Stale same-day detailed source", client.user)
        self.assertNotIn("Internal context", client.user)
        written = json.loads((output_dir / "2026-06-14_narrative_brief.json").read_text(encoding="utf-8"))
        self.assertFalse(written["metadata"]["enrichment_used"])

    def test_pipeline_narrative_respects_enrichment_opt_out_with_disk_sources(self) -> None:
        output_dir = TEMP_ROOT / uuid.uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "2026-06-14_general_brief.json").write_text(
            '{"title":"Daily Brief","lead":"A policy story matters."}',
            encoding="utf-8",
        )
        (output_dir / "2026-06-14_enrichment.json").write_text(
            json.dumps(
                {
                    "schema_version": "enrichment_output.v1",
                    "date": "2026-06-14",
                    "story_threads": [
                        {
                            "story_title": "Policy context",
                            "internal_articles": [
                                {"title": "Internal context", "summary": "Useful background."}
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        client = FakeAIClient()
        orchestrator = FakeOrchestrator(output_dir, client)

        result = run_narrative_brief(
            orchestrator,
            outputs=[],
            date="2026-06-14",
            use_enrichment=False,
        )

        self.assertIsNotNone(result)
        self.assertNotIn("Internal context", client.user)
        written = json.loads((output_dir / "2026-06-14_narrative_brief.json").read_text(encoding="utf-8"))
        self.assertFalse(written["metadata"]["enrichment_used"])


if __name__ == "__main__":
    unittest.main()
