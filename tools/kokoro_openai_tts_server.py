from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import argparse
import json
import math
import os
import struct
import wave


SAMPLE_RATE = 24000
DEFAULT_MODEL_ID = os.environ.get("KOKORO_MODEL_ID", "hexgrad/Kokoro-82M")
DEFAULT_LANG_CODE = os.environ.get("KOKORO_LANG_CODE", "a")
DEFAULT_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")

_pipeline = None
_pipeline_key = None


def get_pipeline(lang_code: str, model_id: str):
    global _pipeline, _pipeline_key
    key = (lang_code, model_id)
    if _pipeline is None or _pipeline_key != key:
        from kokoro import KPipeline

        _pipeline = KPipeline(lang_code=lang_code, repo_id=model_id)
        _pipeline_key = key
    return _pipeline


def synthesize_wav(text: str, *, voice: str, speed: float, lang_code: str, model_id: str) -> bytes:
    pipeline = get_pipeline(lang_code, model_id)
    chunks = []
    for _graphemes, _phonemes, audio in pipeline(text, voice=voice, speed=speed, split_pattern=None):
        chunks.append(audio)
    if not chunks:
        raise RuntimeError("Kokoro returned no audio.")
    return wav_bytes(chunks)


def wav_bytes(audio_chunks) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        batch = []
        for audio in audio_chunks:
            for sample in flatten_samples(audio):
                batch.append(pcm16(sample))
                if len(batch) >= 8192:
                    handle.writeframes(struct.pack(f"<{len(batch)}h", *batch))
                    batch.clear()
        if batch:
            handle.writeframes(struct.pack(f"<{len(batch)}h", *batch))
    return output.getvalue()


def flatten_samples(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_samples(item)
        return
    yield value


def pcm16(value) -> int:
    try:
        sample = float(value)
    except (TypeError, ValueError):
        return 0
    if math.isnan(sample) or math.isinf(sample):
        return 0
    if -1.0 <= sample <= 1.0:
        sample *= 32767.0
    return max(-32768, min(32767, int(round(sample))))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self.send_json({"object": "list", "data": [{"id": "kokoro", "object": "model"}]})
            return
        self.send_error(404)

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/audio/speech":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = str(payload.get("input") or "").strip()
            if not text:
                self.send_error(400, "Missing input")
                return
            response_format = str(payload.get("response_format") or "wav").lower()
            if response_format != "wav":
                self.send_error(400, "Only wav response_format is supported")
                return
            model = str(payload.get("model") or "kokoro")
            model_id = model if "/" in model else DEFAULT_MODEL_ID
            audio = synthesize_wav(
                text,
                voice=str(payload.get("voice") or DEFAULT_VOICE),
                speed=float(payload.get("speed") or 1.0),
                lang_code=str(payload.get("lang_code") or DEFAULT_LANG_CODE),
                model_id=model_id,
            )
        except Exception as exc:
            self.send_error(500, f"{type(exc).__name__}: {exc}")
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)

    def send_json(self, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8880)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Kokoro TTS server: http://{args.host}:{args.port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
