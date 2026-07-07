# TTS Audio

TTS is optional and disabled by default. The first backend is Kokoro.

Kokoro currently supports Python 3.10 through 3.12. If your base environment is Python 3.13 or newer, make a small Python 3.12 environment for TTS.

Install optional audio dependencies:

```bash
conda create -n mydailynews-tts python=3.12
conda activate mydailynews-tts
python -m pip install -r requirements.txt
```

Enable the config block:

```json
{
  "tts": {
    "enabled": true,
    "backend": "kokoro",
    "model_id": "hexgrad/Kokoro-82M",
    "voice": "af_heart",
    "lang_code": "a",
    "speed": 1.0,
    "max_chunk_chars": 1200
  }
}
```

Run against any saved Markdown brief:

```bash
python main.py --module tts --markdown-path output/YYYY-MM-DD_general_brief.md
```

If `--markdown-path` is omitted, TTS falls back to `output/YYYY-MM-DD_narrative_brief.md` for the requested `--date`.

To include audio in the default series, add `tts` after `narrative_brief`:

```json
{
  "pipeline": {
    "default_series": ["briefs", "enrichment", "narrative_brief", "tts"]
  }
}
```

Outputs:

```text
output/<input-markdown-stem>.wav
output/<input-markdown-stem>_audio.json
```

The module consumes narrative Markdown, removes metadata, reference/source sections, Markdown link URLs, and plain URLs, then chunks paragraphs for Kokoro. It does not add SSML, voice cloning, streaming, or MP3 export.
