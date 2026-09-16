# Plex Integration Plan for AIMTV

## Goal
Enable AIMTV to source music tracks and metadata from a Plex Media Server, allowing users to generate AI music videos from their Plex library without manually maintaining a separate Airadio-style folder.

## Overview of Changes

1. **New Dependency**
   - Add `plexapi` and optionally `lyricsgenius` (for fetching lyrics when not embedded) to `pyproject.toml`.

2. **Playlist Source Abstraction**
   - Refactor the playlist discovery to support multiple sources: `AiradioPlaylist` (existing) and `PlexPlaylist` (new).
   - Introduce a base class or protocol `PlaylistSource` with a method `iter_tracks()` yielding `(audio_path, lyrics_path, metadata)`.
   - The CLI will select the source based on flags (`--plex-token`, `--plex-url`, `--plex-section`) or fall back to Airadio.

3. **Metadata Extraction**
   - For each track, obtain:
     - Local or temporary audio file path (if Plex reports direct file access, use it; otherwise download to a temporary cache).
     - Lyrics: attempt to read embedded lyrics (ID3 `USLT`), then fall back to an online lookup via `lyricsgenius` (Genius API) or another provider.
     - Title, artist, album (for logging and display).

4. **Caching Strategy**
   - Introduce a cache directory (`~/.cache/aimtv-plex/`) where streamed audio and fetched lyrics are stored.
   - Use checksums or timestamps to avoid re-downloading unchanged files.
   - Provide a CLI command `aimtv plex-cache` to list, clear, or pre-warm the cache.

5. **Interstitials**
   - Keep the existing interstitial mechanism (ads/station-id) unchanged; users can still maintain a local folder for these.
   - Optionally allow specifying a Plex playlist or folder for interstitials (future work).

6. **CLI Changes**
   - New optional arguments:
     - `--plex-token STRING`   Plex token (can also be read from `PLEX_TOKEN` env).
     - `--plex-url STRING`     Base URL of Plex server (default: `http://localhost:32400`).
     - `--plex-section STRING` Library section name containing music (e.g., "Music").
   - If any of the above are provided, AIMTV uses the Plex source; otherwise uses Airadio.
   - Add subcommand `aimtv plex-cache [clear|info|prefetch]`.

7. **Testing**
   - Add unit tests for `PlexPlaylist` using mock `plexapi` objects.
   - Ensure existing Airadio tests still pass.

8. **Documentation**
   - Update `README.md` with a section "Using a Plex library".
   - Document environment variables and cache location.

## Implementation Steps (ordered)

1. Add dependencies to `pyproject.toml`.
2. Create `src/aimtv/plex_playlist.py` implementing `PlexPlaylistSource`.
3. Modify `src/aimtv/playlist.py` to export a common interface and allow source selection.
4. Update `src/aimtv/cli.py` to parse new flags and instantiate the appropriate source.
5. Implement caching helper in `src/aimtv/paths.py` or new `src/aimtv/cache.py`.
6. Add lyric extraction/fetching utilities in `src/aimtv/lyrics.py`.
7. Write unit tests in `tests/test_plex_playlist.py`.
8. Update `README.md`.
9. Add optional `aimtv plex-cache` subcommand.
10. Final verification and commit.

## License
This project is licensed under the MIT License – see LICENSE file.