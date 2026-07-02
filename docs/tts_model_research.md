# TTS Model Research For MyDailyNews

Date: 2026-07-02

## Recommendation

Use **Kokoro-82M** as the first TTS backend for MyDailyNews.

Why:

- MyDailyNews needs a clear single-narrator news brief, not full voice cloning.
- Kokoro is small enough to be practical for a local-first app: 82M parameters.
- The model card lists an Apache 2.0 license, which is much easier to ship around than GPL or noncommercial model weights.
- The Python usage path is simple: install `kokoro` and `soundfile`, create a pipeline, synthesize text, write WAV.
- It fits the existing architecture: consume `output/YYYY-MM-DD_narrative_brief.md`, produce `output/YYYY-MM-DD_narrative_brief.wav`, and leave the analysis pipeline untouched.

The shortest useful implementation is an optional post-brief module, disabled by default, that runs after `narrative_brief`.

## MyDailyNews Fit

Current repo shape already points to the right place:

- `narrative_briefing` writes the human-readable Markdown that should be spoken.
- `docs/configuration.md` already says narrative generation intentionally avoids SSML and that a future TTS-prep stage should consume narrative Markdown.
- The module series already supports post-brief modules: `briefs -> enrichment -> narrative_brief`.
- Local model download and ignored output conventions already exist: `models/`, `output/`, config examples, and autoconfig.

Do not wire TTS into headline scoring, enrichment, evidence, or final brief synthesis. It should be a final consumer of existing Markdown.

## Model Ranking

| Rank | Model | License | Best For | MyDailyNews Fit | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | Kokoro-82M | Apache 2.0 | Local narration, simple deployment, low compute | Best first choice | Strong quality/size tradeoff and minimal feature surface. |
| 2 | Chatterbox Turbo / Multilingual V3 | MIT | Voice cloning, expressive narration, multilingual work | Strong second choice | More powerful, but heavier and more voice-cloning oriented than needed for first pass. |
| 3 | Piper | GPL-3.0-or-later in the active package/fork | Fast offline CPU TTS | Good fallback, license caveat | Operationally simple, but GPL is a poor default for a permissive MIT repo. |
| 4 | ZONOS2 8B | Apache 2.0 | Highest-end naturalness and voice cloning | Future high-quality path | Likely too heavy for the default local Windows-friendly path. |
| 5 | Dia / Dia2 | Apache 2.0 | Dialogue, two-speaker scripts, expressive nonverbal audio | Niche | Great for dialogue, unnecessary for a single daily news narrator. |
| 6 | Coqui XTTS-v2 | Coqui Public Model License | Multilingual voice cloning | Avoid as default | Good feature set, but license is less clean for this repo. |
| 7 | F5-TTS | Code MIT, pretrained weights CC-BY-NC | Research and noncommercial voice cloning | Avoid as default | Noncommercial pretrained weights are a bad default. |

## Candidate Notes

### Kokoro-82M

Source: https://huggingface.co/hexgrad/Kokoro-82M

Observed facts:

- Hugging Face lists it as text-to-speech, English, Apache 2.0.
- The model card describes Kokoro as an open-weight 82M parameter TTS model.
- The usage example installs `kokoro>=0.9.2` and `soundfile`, then writes 24 kHz WAV output.
- The model card says the weights are Apache-licensed and suitable for real deployments.

Why it should be first:

- It is the least invasive path that still sounds modern.
- No voice-cloning policy surface is needed for the first version.
- It can live behind an optional dependency file instead of bloating the base install.
- Its default voices are enough for a daily news reader.

Likely implementation dependency:

```text
requirements-tts.txt
kokoro>=0.9.2
soundfile
```

Windows note: validate `espeak-ng` or Kokoro's phonemizer dependency path before promising one-command setup on Windows.

### Chatterbox

Source: https://github.com/resemble-ai/chatterbox

Observed facts:

- GitHub lists the project as MIT licensed.
- The README describes Chatterbox as a family of open-source TTS models.
- Current model options include Turbo at 350M parameters and Multilingual V3 at 500M parameters.
- It supports voice prompts, multilingual generation, and paralinguistic tags.
- The README says generated audio includes an imperceptible neural watermark.

Why it is not first:

- Voice cloning and paralinguistic controls are more than MyDailyNews needs for the first cut.
- It brings a larger PyTorch/audio stack than Kokoro.
- The API expects more choices: device, reference clip, language ID, exaggeration/config weights.

When to choose it:

- The user wants a recognizable house voice.
- The user wants expressive spoken briefings with laughter, pauses, emotional tone, or multilingual coverage.
- GPU availability is acceptable.

### Piper

Sources:

- Archived original: https://github.com/rhasspy/piper
- Active package/fork: https://github.com/OHF-Voice/piper1-gpl
- PyPI: https://pypi.org/project/piper-tts/

Observed facts:

- Original `rhasspy/piper` repo is archived and read-only.
- The active Open Home Foundation fork/package is GPL-3.0.
- PyPI shows `piper-tts` 1.4.2 released on 2026-04-02, with Windows wheels.
- Piper is a fast local neural TTS engine and exposes CLI/API paths.

Why it is not first:

- GPL-3.0 is not a nice default dependency for a repo currently licensed MIT.
- Voice quality is usually more "utility TTS" than modern neural narration.

When to choose it:

- The user wants the most boring offline CLI dependency.
- CPU-only speed matters more than voice quality.
- GPL implications are acceptable because it is optional/local only.

### ZONOS2 8B

Source: https://arxiv.org/abs/2606.24320

Observed facts:

- The June 2026 technical report describes ZONOS2 8B as a state-of-the-art TTS model focused on naturalness, prosody, and voice cloning fidelity.
- It scales from Zonos-v0.1 to 8B total parameters with 900M active parameters.
- The report says model weights and inference code are released under Apache 2.0 on GitHub and Hugging Face.

Why it is not first:

- It is likely overkill for a daily briefing narrator.
- It probably wants a more serious GPU/server setup.
- It adds operational complexity where MyDailyNews currently wants local-first, simple setup.

When to choose it:

- The goal becomes "best possible local voice quality."
- A GPU box is expected.
- A separate TTS server/module boundary is acceptable.

### Dia / Dia2

Source: https://github.com/nari-labs/dia

Observed facts:

- Dia is described as a 1.6B parameter TTS model.
- It focuses on realistic dialogue from transcripts and can condition on audio for tone/emotion.
- The README says CPU support was not yet the path tested; GPU/CUDA is the expected route.
- GitHub lists Apache 2.0.

Why it is not first:

- MyDailyNews is a single-narrator briefing, not a scripted two-speaker show.
- Hardware needs are higher than Kokoro.

When to choose it:

- The product direction changes toward a podcast/dialogue format.
- The brief becomes "anchor plus analyst" or multi-speaker audio.

### Coqui XTTS-v2

Source: https://huggingface.co/coqui/XTTS-v2

Observed facts:

- Hugging Face lists the model under the Coqui Public Model License.
- The model supports 17 languages.
- It supports voice cloning from a short audio clip, cross-language voice cloning, emotion/style transfer, and 24 kHz output.

Why it is not first:

- The CPML is more complicated than Apache/MIT.
- Voice cloning is not needed for the initial MyDailyNews feature.
- Coqui's ecosystem status has changed over time, so relying on it as the default path is riskier than Kokoro.

When to choose it:

- The user explicitly wants multilingual voice cloning and accepts the license.

### F5-TTS

Source: https://github.com/SWivid/F5-TTS

Observed facts:

- The code is MIT licensed.
- The pretrained models are CC-BY-NC because of the training data, according to the README.
- It has a mature install/runtime story with PyTorch, FFmpeg, and GPU instructions.

Why it is not first:

- Noncommercial pretrained weights are a dealbreaker for a default repo recommendation.
- The dependency and runtime surface is larger than needed.

When to choose it:

- Research-only/noncommercial usage.
- The user wants to experiment with voice cloning and flow-matching TTS.

## Proposed Implementation Plan

No code yet. When ready, keep the implementation small.

### Phase 1: Optional Kokoro Module

Add:

- `requirements-tts.txt` with Kokoro-only optional dependencies.
- `mydailynews/tts/kokoro_backend.py`
- `mydailynews/pipeline/tts_module.py`
- `tests/test_tts_module.py`

Config:

```json
{
  "tts": {
    "enabled": false,
    "backend": "kokoro",
    "model_id": "hexgrad/Kokoro-82M",
    "voice": "af_heart",
    "lang_code": "a",
    "speed": 1.0,
    "max_chunk_chars": 1200
  }
}
```

Pipeline:

```text
briefs -> enrichment -> narrative_brief -> tts
```

CLI:

```bash
python main.py --module tts --date YYYY-MM-DD
python main.py --module series
```

Output:

```text
output/YYYY-MM-DD_narrative_brief.wav
output/YYYY-MM-DD_narrative_brief_audio.json
```

### Phase 2: Text Cleanup

Input should be the narrative Markdown, not the structured JSON.

Cleanup rules:

- Remove Markdown heading markers.
- Remove `_Generated:` metadata lines.
- Remove source/reference sections from spoken output by default.
- Remove URLs.
- Preserve paragraph breaks.
- Split into chunks at paragraph boundaries.
- Keep chunks under a conservative character limit.

Do not add SSML first. Plain text is enough until a real need appears.

### Phase 3: Synthesis And Stitching

Kokoro yields per-chunk audio. Write each chunk and stitch WAVs only if their sample rate/channel format matches.

Keep one public function:

```python
def synthesize_narrative_markdown(markdown_path: Path, output_path: Path, config: TTSConfig) -> TTSOutput:
    ...
```

Avoid a provider abstraction until a second backend is actually implemented.

### Phase 4: Tests

Small useful tests:

- Markdown cleanup removes metadata, references, and URLs.
- Chunking keeps all text in order and respects max chunk size.
- Standalone TTS loads same-day narrative Markdown from disk.
- Series-mode TTS uses current-run narrative output, not stale same-day disk fallback.
- Fake synthesizer writes a small valid WAV so tests do not download models.

Do not add model-inference tests to the default suite.

### Phase 5: Docs

Add docs only after implementation:

- `docs/tts.md`
- README command examples
- `config.example.json` optional `tts` section
- Troubleshooting notes for `espeak-ng`, `soundfile`, and Hugging Face cache

## Safety And Product Defaults

Default behavior should be non-cloning single-voice narration.

Do not add voice cloning in the first pass. If added later:

- Require an explicit `voice_clone_reference_path`.
- Document consent expectations.
- Add a warning when reference audio is configured.
- Prefer watermarked models if voice cloning becomes a product feature.

## Lazy Implementation Choice

First build:

```text
Kokoro only, optional dependency, disabled by default, consumes narrative Markdown, writes one WAV.
```

Skipped for now:

- Multi-backend abstraction
- Voice cloning
- SSML
- Streaming playback
- GUI audio player
- MP3 export
- TTS autoconfig

Add those only after the basic WAV output proves useful.

## Source Links

- Kokoro-82M model card: https://huggingface.co/hexgrad/Kokoro-82M
- Chatterbox repository: https://github.com/resemble-ai/chatterbox
- Piper archived repository: https://github.com/rhasspy/piper
- Piper active fork/package repository: https://github.com/OHF-Voice/piper1-gpl
- Piper PyPI package: https://pypi.org/project/piper-tts/
- ZONOS2 technical report: https://arxiv.org/abs/2606.24320
- Dia repository: https://github.com/nari-labs/dia
- Coqui XTTS-v2 model card: https://huggingface.co/coqui/XTTS-v2
- F5-TTS repository: https://github.com/SWivid/F5-TTS
