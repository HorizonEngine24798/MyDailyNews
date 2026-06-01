from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import types
import unittest

# Local test environment may not have third-party dependencies installed.
if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace(RequestException=Exception, get=None, post=None)
if "feedparser" not in sys.modules:
    sys.modules["feedparser"] = types.SimpleNamespace(parse=lambda *_args, **_kwargs: types.SimpleNamespace(entries=[]))
if "trafilatura" not in sys.modules:
    sys.modules["trafilatura"] = types.SimpleNamespace(extract=lambda *_args, **_kwargs: "")

from mydailynews.ai.headline_analyzer import HeadlineAnalyzer
from mydailynews.analysis_pipeline import DeltaExtractor, EvidenceDistiller
from mydailynews.brief import BriefGenerator
from mydailynews.models import (
    DeltaExtractionConfig,
    EvidenceDistillationConfig,
    HeadlineDecision,
    NewsCandidate,
    SelectedArticle,
    TopicConfig,
    UserMemory,
)
from mydailynews.prompt_regression import (
    PROMPT_REGRESSION_SCHEMA_VERSION,
    build_prompt_regression_pack,
)
from mydailynews.utils import utc_now


def _fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "docs" / "evaluation" / "fixtures" / "prompt_regression_pack_v1.json"


def _safe_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


class _PromptClient:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(
            backend="transformers",
            effective_model_label="unit-regression-client",
            response_format="json_object",
        )
        self.max_input_tokens = 8192
        self.max_new_tokens = 1024

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def complete_json(*_args, **_kwargs):
        return {}


class PromptRegressionPackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.fixture = json.loads(_fixture_path().read_text(encoding="utf-8"))
        cls.generated = build_prompt_regression_pack()

    def test_fixture_schema_version(self) -> None:
        self.assertEqual(self.fixture.get("schema_version"), PROMPT_REGRESSION_SCHEMA_VERSION)
        self.assertEqual(self.generated.get("schema_version"), PROMPT_REGRESSION_SCHEMA_VERSION)

    def test_stage_prompts_preserve_required_clauses_and_stay_within_budget(self) -> None:
        expected_stages = self.fixture.get("stages", {})
        generated_stages = self.generated.get("stages", {})
        self.assertIsInstance(expected_stages, dict)
        self.assertIsInstance(generated_stages, dict)

        for stage_name, expected_stage in expected_stages.items():
            self.assertIsInstance(expected_stage, dict)
            self.assertIn(stage_name, generated_stages)
            generated_stage = generated_stages[stage_name]
            self.assertIsInstance(generated_stage, dict)

            expected_prompt = str(expected_stage.get("rendered_prompt", ""))
            generated_prompt = str(generated_stage.get("rendered_prompt", ""))
            self.assertTrue(expected_prompt)
            self.assertTrue(generated_prompt)

            required_clauses = expected_stage.get("required_clauses", [])
            self.assertIsInstance(required_clauses, list)
            for clause_raw in required_clauses:
                clause = str(clause_raw).strip()
                if not clause:
                    continue
                self.assertIn(clause, expected_prompt, msg=f"{stage_name}: fixture missing clause {clause!r}")
                self.assertIn(clause, generated_prompt, msg=f"{stage_name}: generated missing clause {clause!r}")

            max_chars = _safe_int(expected_stage.get("max_chars"), default=0)
            if max_chars > 0:
                self.assertLessEqual(
                    len(generated_prompt),
                    max_chars,
                    msg=f"{stage_name}: prompt chars exceeded max_chars",
                )

            baseline_chars = _safe_int(expected_stage.get("prompt_chars"), default=len(expected_prompt))
            max_char_delta = _safe_int(expected_stage.get("max_char_delta"), default=0)
            if max_char_delta > 0:
                self.assertLessEqual(
                    abs(len(generated_prompt) - baseline_chars),
                    max_char_delta,
                    msg=f"{stage_name}: prompt char delta exceeded tolerance",
                )

    def test_response_shape_fixtures_remain_parser_compatible(self) -> None:
        response_shapes = self.fixture.get("response_shapes", {})
        self.assertIsInstance(response_shapes, dict)
        self.assertIn("headline_analysis", response_shapes)
        self.assertIn("evidence_distillation", response_shapes)
        self.assertIn("delta_extraction", response_shapes)
        self.assertIn("final_brief", response_shapes)

        topics = [TopicConfig(name="AI policy"), TopicConfig(name="Semiconductor supply chain")]
        candidate_a = NewsCandidate(
            id="cand-1",
            source="PolicyWire",
            category="policy",
            title="Candidate A",
            url="https://example.com/cand-1",
            snippet="snippet",
            published_at=utc_now(),
            metadata={"topic_name": "AI policy"},
        )
        candidate_b = NewsCandidate(
            id="cand-2",
            source="SupplyWatch",
            category="industry",
            title="Candidate B",
            url="https://example.com/cand-2",
            snippet="snippet",
            published_at=utc_now(),
            metadata={"topic_name": "Semiconductor supply chain"},
        )

        analyzer = HeadlineAnalyzer(_PromptClient(), batch_size=4)
        analyzer._reset_multifactor_stats()
        parsed = analyzer._parse_batch_result(
            response_shapes["headline_analysis"],
            [candidate_a, candidate_b],
            topics,
            label="prompt_regression_headline",
            batch_index=1,
            total_batches=1,
        )
        self.assertIn("cand-1", parsed)
        self.assertIn("cand-2", parsed)
        self.assertGreaterEqual(parsed["cand-1"].personal_relevance, 0.0)
        self.assertEqual(parsed["cand-2"].reason, "")

        distiller = EvidenceDistiller(
            _PromptClient(),
            EvidenceDistillationConfig(enabled=True),
        )
        evidence_packet = distiller._normalize_result(response_shapes["evidence_distillation"])
        self.assertIn("overview", evidence_packet)
        self.assertIn("story_clusters", evidence_packet)
        self.assertIn("reader_qa", evidence_packet)

        extractor = DeltaExtractor(
            _PromptClient(),
            DeltaExtractionConfig(enabled=True),
        )
        delta_packet = extractor._normalize_result(response_shapes["delta_extraction"])
        self.assertIn("baseline_coverage_note", delta_packet)
        self.assertIn("new", delta_packet)
        self.assertIn("evidence_gaps", delta_packet)

        class _BriefClient(_PromptClient):
            def complete_json(self, *_args, **_kwargs):
                return dict(response_shapes["final_brief"])

        selected = [
            SelectedArticle(
                candidate=candidate_a,
                decision=HeadlineDecision(candidate_id="cand-1", score=8.7, topic="AI policy"),
                article_text="full text",
                extraction_status="ok",
            )
        ]
        brief_generator = BriefGenerator(
            _BriefClient(),
            max_context_chars=600,
            input_token_limit=3000,
            max_new_tokens=512,
        )
        brief = brief_generator.generate(
            selected,
            UserMemory(),
            topics,
            [],
            "Detailed brief.",
            "2026-05-31",
            evidence_packet=evidence_packet,
            delta_packet=delta_packet,
        )
        self.assertIn("topic_reports", brief)
        self.assertIn("knowns", brief)
        self.assertIn("unknowns", brief)
        self.assertIn("watch_signals", brief)
        self.assertTrue(isinstance(brief["knowns"], list))
        self.assertTrue(isinstance(brief["unknowns"], list))
        self.assertTrue(isinstance(brief["watch_signals"], list))

    def test_prompt_regression_tool_verify(self) -> None:
        command = [
            sys.executable,
            str(self.repo_root / "tools" / "prompt_regression_pack.py"),
            "--verify",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            self.fail(
                "prompt_regression_pack.py --verify failed:\n"
                + completed.stdout
                + "\n"
                + completed.stderr
            )


if __name__ == "__main__":
    unittest.main()
