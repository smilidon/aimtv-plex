# AI MTV — Plex Integration Build

**AI Music Television**, packaged as `aimtv`: play songs from your local
**Airadio** library or **Plex** Media Server and paint reactive diffusion video over them.
AI MTV uses feedback img2img rather than conventional audio-to-video generation.

> **This is a community fork** (`smilidon/aimtv-plex`) that adds Plex Media Server support
> to the original [Decentricity/AIMTV](https://github.com/Decentricity/aimtv) project.

---

## Table of Contents

1. [Original AIMTV Features](#original-aimtv-features)
2. [Plex Integration Features](#plex-integration-features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Plex Mode](#plex-mode)
7. [Troubleshooting](#troubleshooting)

---

## Original AIMTV Features

- Plays songs from your local **Airadio** library and paints reactive diffusion video over them.
- Uses feedback img2img rather than conventional audio-to-video generation.
- Requires only an NVIDIA GPU with sufficient VRAM.
- Hash-verifies lyrics and provenance records from Airadio.
- No internet required for core functionality after initial model downloads.
- Fully deterministic with `--seed`.

## Plex Integration Features

- Connect AIMTV directly to your **Plex Media Server** to source music from your library.
- Two modes:
  - **Playlist mode (default):** Generates a review MP4 using your Plex music library.
  - **External Player mode:** Integrates AIMTV as a custom player in Plex for on-demand video generation per track.
- Supports local file access (when Plex media is hosted on the same filesystem) or temporary caching.
- Attempts to extract embedded lyrics (ID3 `USLT`) from tracks.
- Falls back to Genius API lyric lookup when embedded lyrics aren't available (requires `GENIUS_TOKEN`).

---

## Requirements

1. **For Airadio mode (original):**
   - Install and run [Airadio](https://pypi.org/project/airadio/) until you have at least 2 library songs, 1 ad, and 1 station-id interstitial.

2. **For Plex mode (new):**
   - A running Plex Media Server instance.
   - A valid Plex token with access to your music library.
   - (Optional) A Genius API token for lyric fallback (set `GENIUS_TOKEN` environment variable).

3. An NVIDIA GPU with enough free VRAM (AIMTV will check and ask you to free the GPU yourself — it never kills other processes).

---

## Installation

### From PyPI (when available)

With pip:

```bash
pip install aimtv[plex]
```

Or as an isolated CLI with pipx:

```bash
pipx install "aimtv[plex]"
pipx install airadio  # only needed for Airadio library mode
```

### From Source (development)

```bash
git clone https://github.com/smilidon/aimtv-plex.git
cd aimtv-plex
pip install -e ".[plex,dev]"
```

---

## Usage

### Original Airadio Mode

Finite playlist of **2 library songs** plus 1–3 interstitials between them, rendered to a single MP4 (audio + generated video):

```bash
aimtv run --review --yes
# → ~/.local/share/aimtv/output/aimtv-review.mp4
```

Inspect a complete plan without loading the model or using the GPU:

```bash
aimtv plan --seed 42
```

### Plex Mode

#### Option 1: Playlist Generation

Generate a review MP4 from your Plex music library:

```bash
# Using environment variables (recommended for tokens):
export PLEX_URL="http://localhost:32400"
export PLEX_TOKEN="your_plex_token_here"
export PLEX_SECTION="Music"  # your music library section name
export GENIUS_TOKEN="your_genius_api_token"  # optional, for lyric fallback

aimtv run --review --yes \
  --plex-url "$PLEX_URL" \
  --plex-token "$PLEX_TOKEN" \
  --plex-section "$PLEX_SECTION"
# → ~/.local/share/aimtv/output/aimtv-review.mp4
```

Or pass tokens directly on the command line:

```bash
aimtv run --review --yes \
  --plex-url "http://localhost:32400" \
  --plex-token "your_token" \
  --plex-section "Music"
```

> **Security Note:** Passing tokens via command-line arguments may expose them in shell history.
> Use environment variables (`PLEX_TOKEN`, `GENIUS_TOKEN`) instead where possible.

#### Plan Without Rendering

Inspect the planned playlist without loading any models:

```bash
aimtv plan --seed 42 \
  --plex-url "$PLEX_URL" \
  --plex-token "$PLEX_TOKEN" \
  --plex-section "$PLEX_SECTION"
```

---

## Troubleshooting

### Can’t connect to Plex?

- Ensure `PLEX_URL` is reachable from the machine running AIMTV.
- Verify the token has access to the specified library section.
- Check that the Plex Media Server is running and the web interface is accessible at `http://localhost:32400/web`.

### No lyrics found?

- AIMTV first checks for embedded lyrics (ID3 `USLT` tag in MP3 files).
- If not found and a `GENIUS_TOKEN` is provided, it attempts to look up lyrics online.
- If neither is available, an empty `.txt` file is created, and you’ll see a warning during rendering.

### Media not found locally?

- If your Plex media is stored on a different machine, AIMTV will attempt to stream it to a local cache folder.
- For best performance, ensure media files are accessible via local file paths from the machine running AIMTV.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

This is a community fork intended for educational and development purposes. Contributions are welcome via pull requests.

---

## Acknowledgements

- Original AIMTV project: [Decentricity/aimtv](https://github.com/Decentricity/aimtv)
- `plexapi` library: [pkkid/python-plexapi](https://github.com/pkkid/python-plexapi)
- Lyrics lookup: [ Genius API](https://genius.com/api-clients) via `lyricsgenius`<tool_call>docsUrl https://api.duckduckgo.com/?q=genius+api+client&format=json
**Note:** Corrected Genius API link above — the actual endpoint requires registration at [https://genius.com/api-clients](https://genius.com/api-clients).
```

## Doctor

```bash
aimtv doctor
```

## Channel Prompts

The editable JSON bank provides the visual style suffix and legacy random-cutaway prompts:

- **Edit this:** `~/.local/share/aimtv/prompts.json` (preferred; created automatically)
- Package default: `aimtv/data/prompts.json`

Schema: `prompts` (list of channel lines), `subjects` (short subject tags for weighting),
`style_suffix` (appended to every pick). Add or remove strings anytime — no code change.
