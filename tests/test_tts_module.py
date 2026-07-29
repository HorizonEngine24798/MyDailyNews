from __future__ import annotations

from pathlib import Path
import json
import sys
import types
import unittest
from unittest.mock import patch
import uuid
import wave

from mydailynews.app.models import AppConfig, TTSConfig, TTSOutput
from mydailynews.diagnostics.debug import DebugLogger
from mydailynews.pipeline.tts_module import run_tts
from mydailynews.tts.kokoro_backend import (
    _iter_audio_chunks,
    _synthesize_audio_chunks,
    _write_wav,
    speech_text_from_markdown,
    split_text_chunks,
    synthesize_narrative_markdown,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = REPO_ROOT / ".codex_tmp_test" / "tts"


class FakeReporter:
    def phase(self, message: str) -> None:
        return None


class FakeOrchestrator:
    def __init__(self, output_dir: Path) -> None:
        self.config = AppConfig(output_dir=str(output_dir), tts=TTSConfig(enabled=True))
        self.warnings: list[str] = []
        self.reporter = FakeReporter()
        self.debug = DebugLogger(False)
        self.artifacts: list[dict] = []

    def _stage_payload(self, *, stage: str, brief_name: str, summary: dict, next_stage_input: dict) -> dict:
        return {"stage": stage, "brief_name": brief_name, "summary": summary, "next_stage_input": next_stage_input}

    def _record_stage_artifact(self, *, stage: str, brief_name: str, payload: dict) -> None:
        self.artifacts.append({"stage": stage, "brief_name": brief_name, "payload": payload})


class TTSModuleTests(unittest.TestCase):
    def _temp_dir(self) -> Path:
        path = TEMP_ROOT / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path

    def test_markdown_cleanup_removes_metadata_references_and_urls(self) -> None:
        markdown = """# Daily Brief

_Generated: 2026-06-14T00:00:00+00:00_
_Source briefs: general, detailed_

Good morning. Read the [policy memo](https://example.test/memo).

## Update

- A useful point at https://example.test/source.

## References

- Source link should not be spoken.

## Closing

That is the brief.
"""

        text = speech_text_from_markdown(markdown)

        self.assertIn("Daily Brief", text)
        self.assertIn("Good morning. Read the policy memo.", text)
        self.assertIn("Update", text)
        self.assertIn("A useful point at", text)
        self.assertIn("Closing", text)
        self.assertNotIn("Generated:", text)
        self.assertNotIn("Source briefs:", text)
        self.assertNotIn("https://", text)
        self.assertNotIn("Source link should not be spoken", text)

    def test_markdown_cleanup_removes_claim_markers(self) -> None:
        self.assertEqual(speech_text_from_markdown("# Brief\n\nThe date is confirmed. <<12>>"), "Brief\n\nThe date is confirmed.")

    def test_chunking_keeps_text_order_under_limit(self) -> None:
        text = "First sentence.\n\nSecond sentence has enough words to need its own chunk.\n\nThird sentence."

        chunks = split_text_chunks(text, 40)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 40 for chunk in chunks))
        self.assertEqual(
            " ".join(" ".join(chunks).split()),
            "First sentence. Second sentence has enough words to need its own chunk. Third sentence.",
        )

    def test_synthesize_narrative_markdown_writes_valid_wav_with_fake_audio(self) -> None:
        output_dir = self._temp_dir()
        markdown_path = output_dir / "2026-06-14_narrative_brief.md"
        wav_path = output_dir / "2026-06-14_narrative_brief.wav"
        markdown_path.write_text("# Brief\n\nHello world.", encoding="utf-8")

        with patch(
            "mydailynews.tts.kokoro_backend._iter_audio_chunks",
            return_value=iter([[0.0, 0.5, -0.5, 0.0]]),
        ):
            output = synthesize_narrative_markdown(markdown_path, wav_path, TTSConfig(enabled=True))

        self.assertTrue(wav_path.exists())
        with wave.open(str(wav_path), "rb") as handle:
            self.assertEqual(handle.getframerate(), 24000)
            self.assertEqual(handle.getnchannels(), 1)
            self.assertGreater(handle.getnframes(), 0)
        payload = json.loads(Path(output.json_path).read_text(encoding="utf-8"))
        self.assertEqual(payload["chunk_count"], output.chunk_count)
        self.assertEqual(payload["voice"], "af_heart")

    def test_wav_writer_moves_tensor_audio_to_numpy(self) -> None:
        calls: list[object] = []

        class FakeAudio:
            def detach(self):
                calls.append("detach")
                return self

            def cpu(self):
                calls.append("cpu")
                return self

            def numpy(self):
                calls.append("numpy")
                return [0.0, 0.5]

        class FakeSoundFile:
            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def write(self, audio: object) -> None:
                calls.append(audio)

        with patch("mydailynews.tts.kokoro_backend.sf.SoundFile", return_value=FakeSoundFile()):
            _write_wav(self._temp_dir() / "audio.wav", 24000, [FakeAudio()])

        self.assertEqual(calls, ["detach", "cpu", "numpy", [0.0, 0.5]])

    def test_kokoro_pipeline_uses_configured_model_id(self) -> None:
        calls: dict[str, object] = {}

        class FakePipeline:
            def __init__(self, **kwargs: object) -> None:
                calls.update(kwargs)

            def __call__(self, *_args: object, **_kwargs: object):
                return [("g", "p", [0.0])]

        fake_kokoro = types.ModuleType("kokoro")
        fake_kokoro.KPipeline = FakePipeline

        with patch.dict(sys.modules, {"kokoro": fake_kokoro}):
            audio_chunks = list(
                _iter_audio_chunks(["Hello."], TTSConfig(enabled=True, model_id="hexgrad/Kokoro-82M"))
            )

        self.assertEqual(calls["repo_id"], "hexgrad/Kokoro-82M")
        self.assertEqual(len(audio_chunks), 1)

    def test_kokoro_missing_runtime_file_has_setup_hint(self) -> None:
        class FakePipeline:
            def __init__(self, **_kwargs: object) -> None:
                raise FileNotFoundError(2, "No such file or directory", "espeak-ng")

        fake_kokoro = types.ModuleType("kokoro")
        fake_kokoro.KPipeline = FakePipeline

        with patch.dict(sys.modules, {"kokoro": fake_kokoro}):
            with self.assertRaisesRegex(RuntimeError, "espeak-ng is installed and on PATH"):
                _synthesize_audio_chunks(["Hello."], TTSConfig(enabled=True))

    def test_standalone_tts_loads_same_day_narrative_markdown_from_disk(self) -> None:
        output_dir = self._temp_dir()
        markdown_path = output_dir / "2026-06-14_narrative_brief.md"
        markdown_path.write_text("# Disk narrative\n\nHello.", encoding="utf-8")
        orchestrator = FakeOrchestrator(output_dir)
        calls: dict[str, Path] = {}

        def fake_synthesize(markdown_path: Path, output_path: Path, config: TTSConfig) -> TTSOutput:
            calls["markdown_path"] = markdown_path
            return TTSOutput(
                name="tts",
                wav_path=str(output_path),
                json_path=str(output_path.with_name(f"{output_path.stem}_audio.json")),
                markdown_path=str(markdown_path),
                voice=config.voice,
                chunk_count=1,
            )

        with patch("mydailynews.pipeline.tts_module.synthesize_narrative_markdown", fake_synthesize):
            output = run_tts(orchestrator, date="2026-06-14")

        self.assertIsNotNone(output)
        self.assertEqual(calls["markdown_path"], markdown_path)
        self.assertEqual(output.markdown_path, str(markdown_path))

    def test_standalone_tts_uses_explicit_general_markdown_path(self) -> None:
        output_dir = self._temp_dir()
        markdown_path = output_dir / "2026-06-14_general_brief.md"
        markdown_path.write_text("# General brief\n\nHello.", encoding="utf-8")
        orchestrator = FakeOrchestrator(output_dir)
        calls: dict[str, Path] = {}

        def fake_synthesize(markdown_path: Path, output_path: Path, config: TTSConfig) -> TTSOutput:
            calls["markdown_path"] = markdown_path
            calls["output_path"] = output_path
            return TTSOutput(
                name="tts",
                wav_path=str(output_path),
                json_path=str(output_path.with_name(f"{output_path.stem}_audio.json")),
                markdown_path=str(markdown_path),
                voice=config.voice,
                chunk_count=1,
            )

        with patch("mydailynews.pipeline.tts_module.synthesize_narrative_markdown", fake_synthesize):
            output = run_tts(orchestrator, markdown_path=str(markdown_path))

        self.assertIsNotNone(output)
        self.assertEqual(calls["markdown_path"], markdown_path)
        self.assertEqual(calls["output_path"], output_dir / "2026-06-14_general_brief.wav")

    def test_series_tts_uses_current_run_markdown_without_disk_fallback(self) -> None:
        output_dir = self._temp_dir()
        current_path = output_dir / "current_narrative.md"
        stale_path = output_dir / "2026-06-14_narrative_brief.md"
        current_path.write_text("# Current narrative\n\nFresh.", encoding="utf-8")
        stale_path.write_text("# Stale narrative\n\nOld.", encoding="utf-8")
        orchestrator = FakeOrchestrator(output_dir)
        calls: dict[str, Path] = {}

        def fake_synthesize(markdown_path: Path, output_path: Path, config: TTSConfig) -> TTSOutput:
            calls["markdown_path"] = markdown_path
            return TTSOutput(
                name="tts",
                wav_path=str(output_path),
                json_path=str(output_path.with_name(f"{output_path.stem}_audio.json")),
                markdown_path=str(markdown_path),
                voice=config.voice,
                chunk_count=1,
            )

        with patch("mydailynews.pipeline.tts_module.synthesize_narrative_markdown", fake_synthesize):
            output = run_tts(
                orchestrator,
                date="2026-06-14",
                markdown_path=str(current_path),
                allow_disk_fallback=False,
            )

        self.assertIsNotNone(output)
        self.assertEqual(calls["markdown_path"], current_path)
        self.assertNotEqual(calls["markdown_path"], stale_path)


if __name__ == "__main__":
    unittest.main()
