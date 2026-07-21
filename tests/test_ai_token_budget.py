from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from mydailynews.ai.base import AIJsonError
from mydailynews.ai.headline_analyzer import HeadlineAnalyzer
from mydailynews.ai.llama_cpp_server_client import LlamaCppServerClient
from mydailynews.ai.token_budget import resolve_client_token_budget, resolve_token_budget
from mydailynews.app.models import AIConfig
from mydailynews.briefing.generator import BriefGenerator
from mydailynews.briefing.narrative import NarrativeBriefGenerator


class _RecordingDebug:
    def __init__(self) -> None:
        self.logs: list[tuple[str, str, dict]] = []
        self.ai_records: list[dict] = []

    def log(self, event: str, message: str = "", **fields) -> None:
        self.logs.append((event, message, fields))

    def record_ai(self, **fields) -> None:
        self.ai_records.append(fields)


def _response(content: str, finish_reason: str, *, usage: dict | None = None, timings: dict | None = None) -> Mock:
    response = Mock(status_code=200, text="")
    payload = {
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
    }
    if usage is not None:
        payload["usage"] = usage
    if timings is not None:
        payload["timings"] = timings
    response.json.return_value = payload
    return response


class TokenBudgetTests(unittest.TestCase):
    def test_inherits_existing_limits_and_clamps_to_context(self) -> None:
        unchanged = resolve_token_budget(
            context_tokens=4096,
            max_input_tokens=3000,
            max_output_tokens=768,
        )
        self.assertEqual((unchanged.input_tokens, unchanged.output_tokens), (3000, 768))

        clamped = resolve_token_budget(
            context_tokens=2048,
            max_input_tokens=1800,
            max_output_tokens=1000,
        )
        self.assertEqual(clamped.input_tokens + clamped.output_tokens + clamped.reserve_tokens, 2048)

    def test_rejects_nonpositive_values(self) -> None:
        valid = {
            "context_tokens": 4096,
            "max_input_tokens": 3000,
            "max_output_tokens": 768,
        }
        for field, value in (
            ("context_tokens", 0),
            ("max_input_tokens", 0),
            ("max_output_tokens", -1),
            ("input_tokens", 0),
            ("output_tokens", 0),
            ("reserve_tokens", 0),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_token_budget(**(valid | {field: value}))

    def test_stage_preferences_are_clamped_to_client_context(self) -> None:
        client = Mock(
            max_input_tokens=3000,
            max_new_tokens=2000,
            config=AIConfig(context_window_tokens=4096),
        )

        budget = resolve_client_token_budget(client, input_tokens=2800, output_tokens=1500)

        self.assertEqual((budget.input_tokens, budget.output_tokens, budget.reserve_tokens), (2340, 1500, 256))

    def test_remaining_stage_fitters_use_the_resolved_budget(self) -> None:
        client = Mock(
            max_input_tokens=3000,
            max_new_tokens=2000,
            config=AIConfig(context_window_tokens=4096),
        )

        headline = HeadlineAnalyzer(client, batch_size=10, input_token_limit=3000, max_new_tokens=1200)
        final_brief = BriefGenerator(client, 1000, input_token_limit=3000, max_new_tokens=1200)
        narrative = NarrativeBriefGenerator(client, input_token_limit=3000, max_new_tokens=1200)

        self.assertEqual((headline._headline_batch_budget(10).input_tokens, headline._headline_batch_budget(10).output_tokens), (2640, 1200))
        self.assertEqual(final_brief._prompt_budget_tokens(), 2508)
        self.assertEqual(narrative.target_words, 708)


class LlamaCppBudgetTests(unittest.TestCase):
    def _client(self, debug: _RecordingDebug | None = None) -> LlamaCppServerClient:
        return LlamaCppServerClient(
            AIConfig(
                context_window_tokens=4096,
                max_input_tokens=3000,
                max_new_tokens=2000,
                json_retries=1,
                manage_server=False,
            ),
            debug=debug,
        )

    def test_parseable_length_response_retries_and_keeps_first_as_fallback(self) -> None:
        client = self._client()
        first = _response('{"items": [1]}', "length", usage={"prompt_tokens": 100, "completion_tokens": 500})
        invalid_retry = _response("not json", "stop")

        with patch(
            "mydailynews.ai.llama_cpp_server_client.requests.post",
            side_effect=[first, invalid_retry],
        ) as post:
            result = client.complete_json("system", "user " * 5000, max_new_tokens=500)

        self.assertEqual(result, {"items": [1]})
        payloads = [call.kwargs["json"] for call in post.call_args_list]
        self.assertEqual([payload["max_tokens"] for payload in payloads], [500, 1000])
        self.assertIn("hit the output token limit", payloads[1]["messages"][1]["content"])
        for payload in payloads:
            prompt_tokens = client._estimate_chat_input_tokens(
                payload["messages"][0]["content"], payload["messages"][1]["content"]
            )
            self.assertLessEqual(prompt_tokens + payload["max_tokens"] + 256, 4096)

    def test_uses_actual_usage_and_exposes_server_timings(self) -> None:
        debug = _RecordingDebug()
        client = self._client(debug)
        first = _response(
            '{"value": "first"}',
            "length",
            usage={"prompt_tokens": 111, "completion_tokens": 40, "total_tokens": 151},
            timings={"predicted_per_second": 12.5},
        )
        second = _response(
            '{"value": "second"}',
            "stop",
            usage={"prompt_tokens": 222, "completion_tokens": 55, "total_tokens": 277},
            timings={"predicted_per_second": 13.5},
        )

        with patch("mydailynews.ai.llama_cpp_server_client.requests.post", side_effect=[first, second]):
            result = client.complete_json("system", "user", max_new_tokens=500)

        self.assertEqual(result, {"value": "second"})
        self.assertEqual(debug.ai_records[-1]["input_tokens"], 222)
        self.assertEqual(debug.ai_records[-1]["output_tokens"], 55)
        self.assertFalse(debug.ai_records[-1]["estimated"])
        response_log = [fields for event, _, fields in debug.logs if event == "ai.response"][-1]
        self.assertEqual(response_log["finish_reason"], "stop")
        self.assertEqual(response_log["timings"]["predicted_per_second"], 13.5)

    def test_does_not_retry_when_context_has_no_more_output_headroom(self) -> None:
        client = self._client()
        client.config.context_window_tokens = 1024
        client.config.max_input_tokens = 700
        client.config.max_new_tokens = 260
        response = _response('{"value": 1}', "length")

        with patch("mydailynews.ai.llama_cpp_server_client.requests.post", return_value=response) as post:
            result = client.complete_json("system", "user", max_new_tokens=260)

        self.assertEqual(result, {"value": 1})
        post.assert_called_once()

    def test_length_retry_is_extra_but_invalid_json_respects_zero_retries(self) -> None:
        client = self._client()
        client.config.json_retries = 0
        invalid = _response("not json", "stop")

        with patch("mydailynews.ai.llama_cpp_server_client.requests.post", return_value=invalid) as post, patch.object(
            client, "_write_invalid_json_artifacts", return_value=("", "")
        ):
            with self.assertRaisesRegex(AIJsonError, "did not return valid JSON"):
                client.complete_json("system", "user")

        post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
