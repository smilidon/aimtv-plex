# AI MTV

**AI Music Television**, packaged as `aimtv`: play songs from your local
**Airadio** library and paint reactive diffusion video over them. AI MTV uses
feedback img2img rather than conventional audio-to-video generation.

## Requirements

1. Install and run [Airadio](https://pypi.org/project/airadio/) until you have at least
   2 library songs, 1 ad, and 1 station-id interstitial.
2. An NVIDIA GPU with enough free VRAM (AI MTV will check and ask you to free the GPU
   yourself — it never kills other processes).

## Install

With pip:

```bash
pip install aimtv
```

Or as an isolated CLI with pipx:

```bash
pipx install aimtv
pipx install airadio
```

AI MTV checks for Airadio at startup. If Airadio is missing it prints both installation
choices. If the library is empty, it asks you to run `airadio` for a bit before trying
again.

## Review build (local testing)

Finite playlist of **2 library songs** plus 1–3 interstitials between them, rendered
to a single MP4 (audio + generated video):

```bash
aimtv run --review --yes
# → ~/.local/share/aimtv/output/aimtv-review.mp4
```

AI MTV reconstructs and hash-verifies each song's exact Airadio lyrics and reads each
interstitial's provenance record. A local Faster-Whisper model recognizes the voice
on CPU, then a monotonic aligner uses those word times only as evidence for where the
verified transcript is being sung. Recognition never replaces or rewrites Airadio's
authoritative words.

A deterministic RAKE/IDF-style phrase ranker supplies contextual imagery without an
LLM, embeddings, or a network service. Interstitial visuals use only phrases from the
verified script and cannot be covered by random footage. Songs deliberately alternate
between well-supported lyric anchors and unrelated dream cutaways; those cutaways are
7–16 seconds long with 3–8 seconds between them by default.

Every render writes an adjacent `*.render.json` manifest tying prompts and timings
back to the verified lyric hash and provenance ID. Inspect a complete plan without
loading the model or using the GPU:

```bash
aimtv plan --seed 42
```

On first run, AI MTV shows every missing model, its destination, and the approximate
download size, then asks once for permission. With consent it installs SD-Turbo
(~6.5 GB) and Faster-Whisper Small (~0.5 GB), showing download progress. Voice timing
runs locally on CPU before the visual model loads on the GPU. Both models and all
per-clip timing results are cached locally. No audio, lyrics, prompts, or inference
requests are sent to a hosted service.

Models can also be installed ahead of time:

```bash
aimtv fetch-models
```

## Doctor

```bash
aimtv doctor
```

## Channel prompts

The editable JSON bank provides the visual style suffix and legacy random-cutaway
prompts:

- **Edit this:** `~/.local/share/aimtv/prompts.json` (preferred; created automatically)
- Package default: `aimtv/data/prompts.json`

Schema: `prompts` (list of channel lines), `subjects` (short subject tags for weighting),
`style_suffix` (appended to every pick). Add or remove strings anytime — no code change.
