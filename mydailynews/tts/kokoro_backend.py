from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import struct
from typing import Any, Iterable, List
import wave

from mydailynews.app.models import TTSConfig, TTSOutput


try:
    import soundfile as sf
except ImportError:
    class _UnavailableSoundFileModule:
        SoundFile = None

    sf = _UnavailableSoundFileModule()


TTS_OUTPUT_SCHEMA_VERSION = "tts_audio.v1"
KOKORO_SAMPLE_RATE = 24000

_HEADING_RE = re.compile(r"^(#{1,6})\s*(.*?)\s*#*\s*$")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_MARKDOWN_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]+\)")
_REFERENCE_HEADING_NAMES = {"references", "reference", "sources", "source", "source links", "citations", "citation"}


def synthesize_narrative_markdown(markdown_path: Path, output_path: Path, config: TTSConfig) -> TTSOutput:
    markdown_path = Path(markdown_path)
    output_path = Path(output_path)
    text = speech_text_from_markdown(markdown_path.read_text(encoding="utf-8-sig"))
    chunks = split_text_chunks(text, int(config.max_chunk_chars))
    if not chunks:
        raise ValueError(f"TTS input has no speakable text: {markdown_path}")

    sample_rate = KOKORO_SAMPLE_RATE
    audio_chunk_count = _write_wav(output_path, sample_rate, _iter_audio_chunks(chunks, config))
    if not audio_chunk_count:
        raise RuntimeError("Kokoro returned no audio chunks.")

    json_path = output_path.with_name(f"{output_path.stem}_audio.json")
    payload = {
        "schema_version": TTS_OUTPUT_SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "backend": config.backend,
        "model_id": config.model_id,
        "voice": config.voice,
        "lang_code": config.lang_code,
        "speed": config.speed,
        "input_markdown_path": str(markdown_path),
        "wav_path": str(output_path),
        "sample_rate": sample_rate,
        "text_chars": len(text),
        "chunk_count": len(chunks),
        "chunks": [{"index": index, "chars": len(chunk)} for index, chunk in enumerate(chunks)],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return TTSOutput(
        name="tts",
        wav_path=str(output_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
        backend=config.backend,
        voice=config.voice,
        chunk_count=len(chunks),
    )


def speech_text_from_markdown(markdown: str) -> str:
    markdown = re.sub(r"<<\d+>>", "", str(markdown or ""))
    lines: List[str] = []
    skip_heading_level: int | None = None
    in_fence = False
    for raw_line in str(markdown or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            if skip_heading_level is not None and level > skip_heading_level:
                continue
            skip_heading_level = None
            heading_text = _clean_inline(heading.group(2))
            if _reference_heading_name(heading_text):
                skip_heading_level = level
                continue
            _append_spoken_line(lines, heading_text)
            continue
        if skip_heading_level is not None:
            continue
        if _metadata_line(line):
            continue
        _append_spoken_line(lines, _clean_inline(line))
    return _join_spoken_lines(lines)


def split_text_chunks(text: str, max_chars: int) -> List[str]:
    max_chars = max(20, int(max_chars))
    chunks: List[str] = []
    current = ""
    for paragraph in [part.strip() for part in re.split(r"\n\s*\n", str(text or "")) if part.strip()]:
        for unit in _paragraph_units(paragraph, max_chars):
            if not current:
                current = unit
            elif len(current) + 2 + len(unit) <= max_chars:
                current = f"{current}\n\n{unit}"
            else:
                chunks.append(current)
                current = unit
    if current:
        chunks.append(current)
    return chunks


def _synthesize_audio_chunks(chunks: List[str], config: TTSConfig) -> tuple[int, List[Any]]:
    return KOKORO_SAMPLE_RATE, list(_iter_audio_chunks(chunks, config))


def _iter_audio_chunks(chunks: List[str], config: TTSConfig) -> Iterable[Any]:
    try:
        from kokoro import KPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Kokoro TTS is not installed. Use Python 3.10-3.12 and install dependencies with requirements.txt."
        ) from exc

    try:
        pipeline = KPipeline(lang_code=config.lang_code, repo_id=config.model_id)
        for chunk in chunks:
            for _graphemes, _phonemes, audio in pipeline(
                chunk,
                voice=config.voice,
                speed=float(config.speed),
                split_pattern=None,
            ):
                yield audio
    except FileNotFoundError as exc:
        missing = f" ({exc.filename})" if exc.filename else ""
        raise RuntimeError(
            f"Kokoro TTS could not find a required runtime file{missing}. "
            "Use Python 3.10-3.12, install requirements.txt, and ensure espeak-ng is installed and on PATH."
        ) from exc


def _write_wav(path: Path, sample_rate: int, audio_chunks: Iterable[Any]) -> int:
    iterator = iter(audio_chunks)
    try:
        first = next(iterator)
    except StopIteration:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    if sf.SoundFile is not None:
        with sf.SoundFile(str(path), mode="w", samplerate=int(sample_rate), channels=1, subtype="PCM_16") as handle:
            handle.write(_audio_array(first))
            count = 1
            for audio in iterator:
                handle.write(_audio_array(audio))
                count += 1
        return count

    # Keep non-TTS workflows and lightweight tests usable without the optional
    # libsndfile dependency. Kokoro's mono float samples map directly to PCM16.
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(_pcm16_bytes(first))
        count = 1
        for audio in iterator:
            handle.writeframes(_pcm16_bytes(audio))
            count += 1
    return count


def _audio_array(audio: Any) -> Any:
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    return audio


def _pcm16_bytes(audio: Any) -> bytes:
    values = _audio_array(audio)
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        values = [values]
    flattened: List[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flattened.extend(float(item) for item in value)
        else:
            flattened.append(float(value))
    samples = [max(-32768, min(32767, int(round(max(-1.0, min(1.0, value)) * 32767.0)))) for value in flattened]
    return struct.pack(f"<{len(samples)}h", *samples) if samples else b""


def _paragraph_units(paragraph: str, max_chars: int) -> List[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    units: List[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        for piece in _hard_wrap(sentence, max_chars):
            if not current:
                current = piece
            elif len(current) + 1 + len(piece) <= max_chars:
                current = f"{current} {piece}"
            else:
                units.append(current)
                current = piece
    if current:
        units.append(current)
    return units


def _hard_wrap(text: str, max_chars: int) -> List[str]:
    pieces: List[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars + 1)
        if split_at < max_chars // 2:
            split_at = max_chars
        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _append_spoken_line(lines: List[str], text: str) -> None:
    if text:
        lines.append(text)
    elif lines and lines[-1]:
        lines.append("")


def _join_spoken_lines(lines: List[str]) -> str:
    output: List[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank and output:
                output.append("")
            previous_blank = True
            continue
        output.append(line)
        previous_blank = False
    return "\n".join(output).strip()


def _clean_inline(text: str) -> str:
    text = _MARKDOWN_LINK_RE.sub(lambda match: match.group(1), str(text or ""))
    text = _URL_RE.sub("", text)
    text = re.sub(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[*_`]+", "", text)
    return " ".join(text.split()).strip()


def _metadata_line(line: str) -> bool:
    text = line.strip().strip("_").strip()
    return text.lower().startswith(("generated:", "source briefs:"))


def _reference_heading_name(text: str) -> bool:
    normalized = " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())
    return normalized in _REFERENCE_HEADING_NAMES
