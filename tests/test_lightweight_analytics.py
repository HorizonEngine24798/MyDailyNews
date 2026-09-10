from __future__ import annotations

from io import StringIO
import unittest

from mydailynews.diagnostics.debug import DebugLogger


class LightweightAnalyticsTests(unittest.TestCase):
    def test_numeric_analytics_remain_enabled_without_debug_events(self) -> None:
        stream = StringIO()
        debug = DebugLogger(False, stream=stream)

        debug.increment("fetch.ok")
        debug.record_ai(
            label="headline scoring",
            status="ok",
            input_tokens=100,
            output_tokens=20,
            retry=True,
            finish_reason="length",
            duration_ms=125.4,
            input_budget_tokens=200,
            output_budget_tokens=64,
        )
        debug.log("fetch", "details that should stay quiet")

        payload = debug.analytics_payload()
        self.assertEqual(payload["counts"]["fetch.ok"], 1)
        self.assertEqual(payload["ai"]["totals"]["finish_reasons"], {"length": 1})
        self.assertEqual(payload["ai"]["totals"]["retries"], 1)
        self.assertEqual(payload["ai"]["totals"]["duration_ms"], 125)
        self.assertEqual(stream.getvalue(), "")

    def test_enabled_events_are_emitted_without_custom_classification_or_throttling(self) -> None:
        stream = StringIO()
        debug = DebugLogger(True, stream=stream)

        debug.log("custom.stage", "progress", count=1)
        debug.log("custom.stage", "progress", count=2)

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("[debug] custom.stage | progress", lines[0])
        self.assertIn('"count": 2', lines[1])

if __name__ == "__main__":
    unittest.main()
