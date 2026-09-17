# AI MTV — Plex Integration Build

**AI Music Television**, packaged as `aimtv`: play songs from your local
**Airadio** library or a **Plex** Media Server music section and paint reactive
diffusion video over them. AI MTV uses feedback img2img rather than conventional
audio-to-video generation.

> **This is a community fork** (`smilidon/aimtv-plex`) that adds Plex Media Server
> support to the original [Decentricity/aimtv](https://github.com/Decentricity/aimtv)
> project. It retains the Airadio playback model and includes the reliability fixes
> described in [PLEX_INTEGRATION.md](PLEX_INTEGRATION.md).

## Requirements

1. Python 3.10 or newer, plus **FFmpeg and ffprobe** on `PATH`. On Debian/Ubuntu:
   `sudo apt-get install ffmpeg`. Git is also needed for the repository installs below.
2. Install and run [Airadio](https://pypi.org/project/airadio/) until you have at least
   2 library songs, 1 ad, and 1 station-id interstitial. Airadio is required in both
   modes: interstitials and provenance verification always come from it. The current
   startup check still requires these assets in Plex mode, even though only the
   Plex songs are selected for playback.
3. An NVIDIA GPU with enough free VRAM (AI MTV will check and ask you to free the GPU
   yourself — it never kills other processes).
4. **Plex mode only:** a reachable Plex Media Server, a Plex token with access to a
   **music** section, and the media files readable on this machine at the paths Plex
   reports (run AI MTV on the Plex host, or mount the media share at the same path).
   Optionally a Genius API token for lyric lookup when tracks have no embedded lyrics.

## Install this fork

Install from this repository to select the Plex fork explicitly. A plain
`pip install aimtv` or `pip install "aimtv[plex]"` selects a package from your package
index, not this GitHub repository.

With pip, preferably inside a virtual environment:

```bash
python -m pip install "aimtv[plex] @ git+https://github.com/smilidon/aimtv-plex.git"
```

Or as an isolated CLI with pipx:

```bash
pipx install --include-deps "aimtv[plex] @ git+https://github.com/smilidon/aimtv-plex.git"
```

`--include-deps` also exposes the Airadio CLI from the same environment. Installing
Airadio into a separate pipx environment does not make its Python modules importable
inside AI MTV's environment.

From source:

```bash
git clone https://github.com/smilidon/aimtv-plex.git
cd aimtv-plex
python -m pip install -e ".[plex,dev]"
```

AI MTV checks for Airadio at startup. If the library is empty, it asks you to run
`airadio` for a bit before trying again.

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

Planning still reads and stitches the selected audio, so it requires local media and
FFmpeg. To omit interstitials from playback, pass both `--interstitial-min 0` and
`--interstitial-max 0`; the startup asset requirement above still applies.

On first run, AI MTV shows every missing model, its destination, and the approximate
download size, then asks once for permission. With consent it installs SD-Turbo
(~6.5 GB) and Faster-Whisper Small (~0.5 GB), showing download progress. Voice timing
runs locally on CPU before the visual model loads on the GPU. Both models and all
per-clip timing results are cached locally. No audio, lyrics, prompts, or inference
requests are sent to a hosted service — except the optional Genius lookup in Plex mode.

Models can also be installed ahead of time:

```bash
aimtv fetch-models
```

## Plex mode

Give `run` or `plan` a Plex server, token and music section and the songs are sampled
from that section instead of the Airadio library. Every Plex connection flag has an
environment variable fallback; use the variables for tokens so they do not appear
in the command's process arguments. Keep token values out of shared logs and shell
history; the exports below contain placeholders only.

```bash
export PLEX_URL="http://localhost:32400"
export PLEX_TOKEN="your_plex_token"
export PLEX_SECTION="Music"              # library section name
export GENIUS_TOKEN="your_genius_token"   # optional lyric fallback

aimtv plan --seed 42          # inspect the playlist without loading models
aimtv run --review --yes      # render
```

The same values can be passed as `--plex-url`, `--plex-token`, `--plex-section` and
`--genius-token`. All three Plex values are required together; a partial set is an
error. Movie and TV sections are rejected with an explicit music-library error.

How it works:

- Up to `max(100, --songs)` tracks are listed from the section (metadata only), then
  `--songs` are sampled with the render seed. Reproducibility assumes the same library
  and returned track order; this is not uniform sampling across a larger library.
- Audio is read straight from the file path Plex reports. Streaming/downloading media
  from Plex is **not** implemented. Mixed sample rates/channel counts are normalized
  before joining audio pieces, preserving their duration and pitch. Matching PCM
  pieces keep the stream-copy path.
- Lyrics are taken from embedded tags (ID3 `USLT`, Vorbis/FLAC `LYRICS`, MP4 `©lyr`),
  then from Genius if `GENIUS_TOKEN` is set. The text is cached under
  `~/.cache/aimtv/plex/<server-id-sha256>/<ratingKey>.lyrics.txt`, with source, file
  identity and lyric hash in an adjacent `<ratingKey>.lyrics.json` file. Override
  the cache root with `AIMTV_CACHE_HOME`. Tokens are not stored in these files.
- Cached lyrics are refreshed when the local audio path, size, modification time or
  track title/artist/album changes, or when the cache is incomplete or corrupt.
  Missing lyrics are retried on later runs, so adding tags or enabling Genius works
  without manual cache removal. Delete a track's `.lyrics.txt` to force a refresh.
  Old flat `<ratingKey>.lyrics.txt` caches are ignored, not migrated or deleted.
- The render manifest records the lyric source (`plex embedded lyrics` or
  `plex genius lyrics`), server-scoped provenance ID and SHA-256 of the text used.
  Unlike Airadio songs, that text is not reconstructed from a catalog — embedded
  tags and Genius are trusted as given. Here, `verified` means sourced and traceable,
  not independently confirmed as the words performed in the recording.
- A track with no lyrics or an unknown lyric source is reported as `UNVERIFIED` and
  `run` refuses to render. Tag the file or provide a Genius token.

### Troubleshooting

- **Can't connect:** check `PLEX_URL` is reachable from this machine and the token is
  valid for that server (`http://<host>:32400/web` should load).
- **Section not found:** `PLEX_SECTION` must match the library name in Plex exactly
  and must identify a music library.
- **"not readable locally":** the path Plex reports for the track does not exist
  here. Run on the Plex host or mount the share at the same path.
- **No lyrics:** tag the file with lyrics, or set `GENIUS_TOKEN`, then retry.
- **Missing ffmpeg/ffprobe:** install the system FFmpeg package and ensure both
  executables are available on `PATH`.

## Doctor

```bash
aimtv doctor
```

This checks Airadio assets and NVIDIA GPU headroom; it is not a live Plex connection
test or a comprehensive dependency check.

## Channel prompts

The editable JSON bank provides the visual style suffix and legacy random-cutaway
prompts:

- **Edit this:** `~/.local/share/aimtv/prompts.json` (preferred; created automatically)
- Package default: `aimtv/data/prompts.json`

Schema: `prompts` (list of channel lines), `subjects` (short subject tags for weighting),
`style_suffix` (appended to every pick). Add or remove strings anytime — no code change.

## Tests

With the development and Plex dependencies installed, run:

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
```

The CI workflow runs the suite on Python 3.10 and 3.12, checks for undefined/unused
Python names with pyflakes, and builds/checks the distributions. Plex/Genius calls
are mocked; audio integration tests use real FFmpeg and synthetic tones. No live
Plex server, credentials, model downloads or GPU are needed for these tests.

## License and acknowledgements

MIT — see [LICENSE](LICENSE). Plex access uses
[python-plexapi](https://github.com/pkkid/python-plexapi); lyric lookup uses
[lyricsgenius](https://github.com/johnwmillr/LyricsGenius) against the
[Genius API](https://genius.com/api-clients).
