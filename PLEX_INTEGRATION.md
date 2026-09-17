# Plex integration — design notes

Status of the Plex source in this fork. User-facing usage lives in `README.md`.

## Implemented

- `aimtv.playlist.PlexConfig` (url, token, section, optional Genius token), built by
  the CLI from `--plex-*` flags with `PLEX_URL` / `PLEX_TOKEN` / `PLEX_SECTION` /
  `GENIUS_TOKEN` fallbacks, and threaded through `render_review_mp4` →
  `build_review_plan` → `build_review_playlist`. Token fields are omitted from the
  configuration's `repr`; this does not protect tokens passed on the command line.
- `aimtv.plex_playlist.PlexPlaylistSource`: accepts music sections, lists up to
  `max(100, song_count)` rating keys, and resolves only the sampled tracks to local
  file paths and cached lyrics.
- Lyrics: embedded tags via mutagen (ID3 `USLT`, Vorbis/FLAC `LYRICS`, MP4 `©lyr`),
  then Genius via `lyricsgenius`. Newlines are normalized before caching/hashing.
- `aimtv.context.resolve_contexts` accepts clips that carry `lyrics_path`; the text is
  hashed and its supported source recorded in the manifest. Empty, unreadable, or
  unknown-source lyrics are `UNVERIFIED`. Source traceability does not independently
  verify whether lyrics match the performance.
- PCM pieces with different sample rates, channel counts or subtypes are normalized
  before concatenation. Compatible pieces keep stream-copy behavior and their PCM
  samples. The Airadio fade and bridge-crossfade rules remain unchanged.
- Zero requested interstitials now produces none. Interstitial candidates are sorted
  before seeded selection so filesystem enumeration order does not change the picks.
- Airadio remains a hard requirement. Its startup asset check still requires two
  library songs, an ad and a station ID in both modes, even when interstitials are
  disabled or Plex supplies the selected songs.

## Cache format and invalidation

Cache root: `~/.cache/aimtv/plex`, or `$AIMTV_CACHE_HOME/plex`.

Each server gets a subdirectory named with the SHA-256 of its Plex
`machineIdentifier`. A track is stored as `<ratingKey>.lyrics.txt` plus
`<ratingKey>.lyrics.json`. Rating keys alone are not global identities: two Plex
servers may assign the same key to different songs. Render provenance IDs therefore
also include the hashed server identity.

The JSON record contains schema version, lyric source and SHA-256, local audio path,
size, nanosecond modification time, and track title/artist/album. Reuse requires a
nonempty lyric file, a supported source, a matching identity and a matching lyric
hash. Changes, missing metadata or corrupt files cause a refresh. File identity is
stat/metadata based, not a full audio-content hash: preserving all those fields while
changing a file requires manual cache invalidation.

Empty results are retried on subsequent runs, including after a transient Genius
failure or enabling a Genius token. Each file is atomically replaced using a unique
temporary file; readers check the lyric hash against its metadata before accepting a
cache hit. Neither Plex nor Genius tokens are written to the cache. Paths and track
metadata are local data and may still be sensitive when sharing cache files.

Old flat `<ratingKey>.lyrics.txt` / `.source` caches are intentionally ignored rather
than migrated, because their server identity is unknown. They are not deleted.
Delete a current track's `.lyrics.txt` file to force a refresh.

## Validation

- Existing `tests/test_plex_playlist.py` covers CLI plumbing and sampled resolution
  with a stub source.
- `tests/test_review_regressions.py` exercises the actual source/cache implementation
  with mocked Plex/Genius requests, cache invalidation/recovery, source validation,
  token repr, real ID3 tags, playlist controls and deterministic interstitial picks.
- Synthetic audio tests exercise real FFmpeg for mixed-format duration/pitch,
  sample-exact compatible concatenation, and song/interstitial timeline alignment.
- CI installs FFmpeg explicitly, runs pyflakes and pytest on Python 3.10 and 3.12,
  and builds/checks wheel and source distributions. No inference models or live
  services are needed for these checks.

## Not implemented / open

- Streaming or downloading media from Plex. Media must be readable on disk at the path
  Plex reports. A download path would need a size/eviction policy for the cache.
- Interstitials from Plex (a playlist or a second section).
- Source-aware Airadio startup asset requirements and a Plex-aware `doctor` command.
- `aimtv plex-cache` management subcommand; today, delete selected cache files manually.
- Uniform seeded sampling across a large section. Sampling is limited to the first
  `max(100, song_count)` tracks Plex returns.
- Independent verification of Genius/embedded lyrics against the recording. The
  existing voice-alignment step supplies timing evidence, not authoritative lyrics.
- Concurrent render isolation. Review plans and videos still share the default
  `review-staging` directory; do not run simultaneous renders with the same `AIMTV_HOME`.
- End-to-end validation against a real Plex server and NVIDIA GPU remains a separate
  operational check; passing the automated suite does not establish it.
