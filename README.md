# AI MTV Plex

**AI Music Television**: select songs from your local **Airadio** library or a
**Plex Media Server** music section and render reactive diffusion video over them.
AI MTV uses feedback img2img, not conventional audio-to-video generation.

**Release: 0.5.0a1 (Alpha 1).** Distribution: `aimtv-plex`. Command and Python imports:
`aimtv`. [Release downloads](https://github.com/smilidon/aimtv-plex/releases/tag/v0.5.0a1)
include a wheel, source distribution and SHA256SUMS.

> This is the community Plex fork of [Decentricity/aimtv](https://github.com/Decentricity/aimtv),
> not an official Plex product or a Plex plugin. It retains the upstream Airadio
> playback model and MIT license. Automated tests do not establish live Plex/GPU
> compatibility; see [release notes](docs/releases/0.5.0a1.md) and
> [integration details and limitations](PLEX_INTEGRATION.md).

## Requirements

- Python 3.10 or newer on Linux, plus **FFmpeg and ffprobe** on `PATH`.
  On Debian/Ubuntu: `sudo apt-get install ffmpeg`.
- **Airadio** installed in the same Python environment. Run it until it has at least
  2 library songs, 1 ad and 1 station-id interstitial. The current startup check
  requires these assets even in Plex mode and when interstitial playback is disabled.
- An NVIDIA GPU with enough free VRAM. AI MTV checks headroom and asks you to stop
  GPU-heavy applications yourself; it never kills other processes.
- For Plex songs: a reachable Plex server, a token with access to a **music** section,
  and files readable locally at the exact paths reported by Plex. Run on the Plex
  host or mount the media share at those paths. Streaming/downloading from Plex is
  not implemented. A Genius API token is optional for lyric lookup.

## Install the versioned release

A fresh virtual environment avoids conflicts with upstream `aimtv` and older fork
installs. Do **not** install `aimtv` and `aimtv-plex` together: both provide the same
Python modules and `aimtv` executable. The project name changed, but your local
`AIMTV_HOME`, `AIRADIO_HOME` and cache locations did not.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "aimtv-plex[plex] @ https://github.com/smilidon/aimtv-plex/releases/download/v0.5.0a1/aimtv_plex-0.5.0a1-py3-none-any.whl"
aimtv --version
# aimtv 0.5.0a1
```

Or install an isolated CLI with pipx:

```bash
pipx install --include-deps "aimtv-plex[plex] @ https://github.com/smilidon/aimtv-plex/releases/download/v0.5.0a1/aimtv_plex-0.5.0a1-py3-none-any.whl"
```

`--include-deps` exposes the Airadio CLI from the same environment. Installing
Airadio in a separate pipx environment does not make its modules importable here.
The wheel contains the application, not FFmpeg, GPU drivers or model weights;
pip installs Python dependencies and models require a separate consented download.

**PyPI is a separate destination.** Use the GitHub wheel above unless the exact
version is present on [PyPI](https://pypi.org/project/aimtv-plex/). After successful
PyPI publication, the equivalent index command is:

```bash
python -m pip install "aimtv-plex[plex]==0.5.0a1"
```

A plain `pip install aimtv` selects upstream, not this fork. For source development:

```bash
git clone https://github.com/smilidon/aimtv-plex.git
cd aimtv-plex
python -m pip install -e ".[plex,dev]"
```

## First render

Populate the Airadio assets, configure Plex below when using it, then inspect a plan:

```bash
aimtv plan --seed 42
aimtv run --review --yes
# ~/.local/share/aimtv/output/aimtv-review.mp4
```

The default finite review contains 2 songs and 1–3 interstitials between them. Both
commands support `--songs`, `--seed`, `--interstitial-min` and `--interstitial-max`.
Set both interstitial bounds to zero to omit them from playback. Planning does not
load GPU models, but it still reads/stitches local audio and requires FFmpeg.

For an opening preview or an explicitly named output:

```bash
aimtv run --review --yes --max-seconds 15 --out "$HOME/aimtv-renders/preview.mp4"
```

On first run AI MTV lists missing models, destinations and approximate download
sizes, then requests consent. `--yes` supplies that consent without prompting.
The SD-Turbo footprint is approximately 6.5 GB and Faster-Whisper Small about 0.5 GB.
Voice timing runs on CPU before the visual model loads on the GPU. Models and timing
results are cached locally. To install models ahead of time:

```bash
aimtv fetch-models
```

No audio, lyrics, prompts or inference requests are sent to a hosted inference
service. The optional Genius lookup sends track title/artist queries to Genius;
model downloads also require network access.

## Plex configuration

Every Plex connection flag has an environment-variable fallback. Variables keep
secrets out of process arguments, but do not paste real tokens into shared logs,
issues or shell history. These examples contain placeholders only:

```bash
export PLEX_URL="http://localhost:32400"
export PLEX_TOKEN="your_plex_token"
export PLEX_SECTION="Music"
export GENIUS_TOKEN="your_genius_token"  # optional

aimtv plan --seed 42
aimtv run --review --yes
```

Flags: `--plex-url`, `--plex-token`, `--plex-section`, `--genius-token`. URL, Plex token
and section are required together; a partial configuration is an error. Movie/TV
sections are rejected. Interstitials and their provenance still come from Airadio.

Up to `max(100, --songs)` track keys are listed before seeded selection; only the
chosen tracks are resolved. This is not uniform sampling over larger libraries.
Reproducibility assumes the same library and returned track order. Interstitial
candidates are sorted before sampling to avoid filesystem-order differences.

### Lyrics, traceability and caching

Plex lyrics come from embedded tags (ID3 `USLT`, Vorbis/FLAC `LYRICS`, MP4 `©lyr`),
then Genius when configured. A missing lyric or unknown source is `UNVERIFIED` and
rendering is refused. Add tags or configure Genius, then retry.

Caches live at `~/.cache/aimtv/plex/<server-id-sha256>/<ratingKey>.lyrics.txt` with
source, file identity and lyric hash in a `.lyrics.json` sidecar. Set
`AIMTV_CACHE_HOME` to change the root. Tokens are not stored in cache files.
Changed paths, file sizes/mtimes or track metadata invalidate a cached entry;
incomplete/corrupt entries are rebuilt and empty results retried. Delete a current
track's `.lyrics.txt` to force refresh. Old flat caches are ignored, not deleted.

Every render writes an adjacent `*.render.json` manifest recording lyric hashes,
sources, prompts and timing evidence. For Plex, `verified` means **sourced and
traceable**, not independently proven to match the sung performance. Airadio song
lyrics are reconstructed from the catalog and hash-verified; interstitial scripts
are resolved from their provenance records.

Faster-Whisper supplies local voice timing evidence, not replacement lyrics. A
local deterministic RAKE/IDF-style phrase ranker supplies imagery without an LLM,
embeddings or hosted service. Interstitial visuals use their verified scripts.
Songs alternate contextual lyric imagery with deliberate dream cutaways (normally
7–16 seconds long, separated by 3–8 seconds). Mixed PCM audio formats are normalized
before joining; compatible PCM retains the sample-preserving stream-copy path.

## Watch the results in Plex (no plugin required)

Plex has removed support for playback through plugins and announced retirement of
the legacy plugin framework. Do not put this Python package in Plex's `Plug-ins`
folder. It is a separate application that reads library metadata and local audio,
then creates MP4 files for Plex to serve.

1. Render outside any watched Plex library, using a distinct `--out` filename.
2. After rendering completes successfully, copy the finished MP4 into a folder
   accessible to Plex. Keep the render manifest for provenance.
3. In Plex Web, add an **Other Videos** library named **AI MTV** pointing to that
   folder, or add the folder to an existing suitable library. Scan library files.
4. Play the resulting video from that library in your Plex clients.

Rendering outside the watched folder prevents Plex from scanning a partially
written MP4. The current release does not add a Plex interface, inject live visuals
into Plex/Plexamp playback, automatically scan Plex, or expose a live TV channel.
A GUI or containerized companion service would be a separate future feature.

Official Plex references: [plugin support](https://support.plex.tv/articles/201053748-overview/),
[framework retirement announcement](https://forums.plex.tv/t/important-information-for-users-running-plex-media-server-on-nvidia-shield-devices/883484),
and [library setup](https://support.plex.tv/articles/200288896-basic-setup-wizard/).

## Troubleshooting and doctor

```bash
aimtv doctor
```

Doctor checks Airadio assets and NVIDIA GPU headroom; it is not a live Plex
connection test or comprehensive dependency check. Check Plex reachability and
credentials for connection failures, and match the music section's name exactly.
A "not readable locally" error means Plex's file paths do not exist on this host.
Missing FFmpeg/ffprobe errors require installing the system FFmpeg package on PATH.

Do not run simultaneous renders with the same `AIMTV_HOME`: review staging paths
are shared. [Known limitations](PLEX_INTEGRATION.md#not-implemented--open) also cover
sampling, lyric trust, missing streaming support and source-aware startup checks.

## Editable prompts

Edit `~/.local/share/aimtv/prompts.json` (created automatically), or use the bundled
`aimtv/data/prompts.json` as a reference. Schema: `prompts` (channel lines),
`subjects` (short subject tags), and `style_suffix` (appended to picks).

## Development and releases

With `.[plex,dev]` and FFmpeg installed:

```bash
python -m pyflakes src/aimtv tests
python -m pytest -q
python -m build
python -m twine check dist/*
```

CI uses Python 3.10 and 3.12. Plex/Genius calls are mocked; audio regressions use
real FFmpeg and synthetic tones. No live credentials, model downloads or GPU are
used. The release workflow also installs the wheel in a fresh environment to check
its metadata, package resources and CLI; this is not full inference validation.

See [RELEASING.md](docs/RELEASING.md) for versioning, GitHub releases and PyPI
Trusted Publishing setup. The version in `src/aimtv/__init__.py` drives package
metadata, `aimtv --version` and render manifests.

## License and acknowledgements

MIT — see [LICENSE](LICENSE). Original AI MTV by Decentricity; Plex community fork
maintained by smilidon. Plex access uses [python-plexapi](https://github.com/pkkid/python-plexapi);
lyric lookup uses [lyricsgenius](https://github.com/johnwmillr/LyricsGenius).
This project is not affiliated with or endorsed by Plex. Use media and lyrics you
have the necessary rights to use; the code license does not license third-party media.
