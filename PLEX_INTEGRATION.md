# Plex integration — design notes

Status of the Plex source in this fork. User-facing usage lives in `README.md`.

## Implemented

- `aimtv.playlist.PlexConfig` (url, token, section, optional Genius token), built by
  the CLI from `--plex-*` flags with `PLEX_URL` / `PLEX_TOKEN` / `PLEX_SECTION` /
  `GENIUS_TOKEN` fallbacks, and threaded through `render_review_mp4` →
  `build_review_plan` → `build_review_playlist`.
- `aimtv.plex_playlist.PlexPlaylistSource`: lists up to 100 rating keys from the
  section, resolves only the sampled tracks to a local file path plus a cached lyrics
  file (`~/.cache/aimtv/plex/<ratingKey>.lyrics.txt` + `.source` marker).
- Lyrics: embedded tags via mutagen (ID3 `USLT`, Vorbis/FLAC `LYRICS`, MP4 `©lyr`),
  then Genius via `lyricsgenius`.
- `aimtv.context.resolve_contexts` accepts clips that carry `lyrics_path`; the text is
  hashed and its source recorded in the manifest. Empty lyrics → `UNVERIFIED`.
- Interstitials, provenance and Airadio preflight are unchanged; Airadio is still a
  hard requirement.
- Tests: `tests/test_plex_playlist.py` (source stubbed; no Plex server needed).

## Not implemented / open

- Streaming or downloading media from Plex. Media must be readable on disk at the path
  Plex reports. A download path would need a size/eviction policy for the cache.
- Interstitials from Plex (a playlist or a second section).
- `aimtv plex-cache` management subcommand; today, delete files under the cache dir.
- Sampling is limited to the first 100 tracks Plex returns. A random-offset query
  would give a fairer sample over very large sections.
- Genius lyrics are trusted as returned; there is no cross-check against the audio
  beyond the existing voice-alignment step.
