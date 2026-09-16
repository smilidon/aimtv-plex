"""Plex playlist source: resolve tracks in a Plex music section to local files + lyrics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from plexapi.exceptions import BadRequest, NotFound, Unauthorized
from plexapi.server import PlexServer

from aimtv.paths import aimtv_cache_dir


def _extract_embedded_lyrics(audio_path: Path) -> str | None:
    """Return unsynchronised lyrics from the file's tags, or None."""
    try:
        from mutagen import File as MutagenFile
        from mutagen.id3 import ID3, USLT
    except ImportError:  # pragma: no cover - optional dependency
        return None

    tags = None
    try:
        audio = MutagenFile(audio_path)
        tags = audio.tags if audio is not None else None
    except Exception:
        tags = None
    if tags is None:
        # Container detection failed; ID3 tags can still be read on their own.
        try:
            tags = ID3(audio_path)
        except Exception:
            return None

    # ID3 (mp3, aiff, wav): USLT frames.
    getall = getattr(tags, "getall", None)
    if callable(getall):
        for frame in getall("USLT"):
            if isinstance(frame, USLT) and str(frame.text).strip():
                return str(frame.text)
        return None
    # Vorbis/FLAC/MP4 style tags expose lyrics under a plain key.
    for key in ("LYRICS", "UNSYNCEDLYRICS", "lyrics", "\xa9lyr"):
        try:
            value = tags.get(key)
        except Exception:
            value = None
        if value:
            text = value[0] if isinstance(value, (list, tuple)) else value
            if str(text).strip():
                return str(text)
    return None


class PlexPlaylistSource:
    """Resolve tracks from a Plex library section to (audio_path, lyrics_path, metadata).

    Audio must be readable on the local filesystem at the path Plex reports for
    the track (same host, or the media share mounted at the same path). Streaming
    or downloading media from Plex is not implemented.

    Lyrics come from embedded tags first, then from Genius when a token is given.
    Both are cached under `cache_dir` keyed by the track's Plex rating key.
    """

    def __init__(
        self,
        baseurl: str,
        token: str,
        section_name: str,
        genius_token: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        try:
            self.plex = PlexServer(baseurl, token)
        except Unauthorized as exc:
            raise ValueError(f"Plex rejected the token for {baseurl}.") from exc
        try:
            self.section = self.plex.library.section(section_name)
        except (NotFound, BadRequest) as exc:
            raise ValueError(f"Plex section '{section_name}' not found or inaccessible.") from exc

        self.genius = None
        if genius_token:
            from lyricsgenius import Genius

            self.genius = Genius(genius_token, remove_section_headers=True)
        self.cache_dir = cache_dir or aimtv_cache_dir() / "plex"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- track discovery -------------------------------------------------

    def track_keys(self, *, limit: int | None = None) -> list[int]:
        """Rating keys of tracks in the section (metadata only; no file access)."""
        kwargs = {"maxresults": limit} if limit else {}
        return [int(track.ratingKey) for track in self.section.searchTracks(**kwargs)]

    def resolve_tracks(self, keys: Iterable[int]) -> Iterator[tuple[Path, Path, dict]]:
        for key in keys:
            yield self._resolve(self.plex.fetchItem(int(key)))

    def iter_tracks(self) -> Iterator[tuple[Path, Path, dict]]:
        """Resolve every track in the section (may be slow for large libraries)."""
        for track in self.section.searchTracks():
            yield self._resolve(track)

    # -- resolution --------------------------------------------------------

    def _resolve(self, track) -> tuple[Path, Path, dict]:
        audio_path = self._local_file_path(track)
        lyrics_path, lyrics_source = self._lyrics_for(track, audio_path)
        metadata = {
            "rating_key": int(track.ratingKey),
            "title": track.title,
            "artist": getattr(track, "grandparentTitle", None),
            "album": getattr(track, "parentTitle", None),
            "lyrics_source": lyrics_source,
        }
        return audio_path, lyrics_path, metadata

    @staticmethod
    def _local_file_path(track) -> Path:
        locations = list(getattr(track, "locations", []) or [])
        for loc in locations:
            path = Path(loc)
            if path.is_file():
                return path
        hint = locations[0] if locations else "<no file path reported by Plex>"
        raise RuntimeError(
            f"Plex track '{track.title}' is not readable locally at {hint}. "
            "AI MTV reads Plex media straight from disk: run it on the Plex host or "
            "mount the media share at the same path Plex uses."
        )

    def _lyrics_cache_path(self, track) -> Path:
        return self.cache_dir / f"{int(track.ratingKey)}.lyrics.txt"

    def _lyrics_for(self, track, audio_path: Path) -> tuple[Path, str | None]:
        """Return (lyrics_path, source). The file is empty when no lyrics were found."""
        cache = self._lyrics_cache_path(track)
        marker = cache.with_suffix(".source")
        if cache.is_file():
            source = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
            return cache, source or None

        lyrics = _extract_embedded_lyrics(audio_path)
        source: str | None = "embedded" if lyrics else None
        if lyrics is None and self.genius is not None:
            lyrics = self._genius_lyrics(track)
            source = "genius" if lyrics else None

        cache.write_text(lyrics or "", encoding="utf-8")
        marker.write_text(source or "", encoding="utf-8")
        return cache, source

    def _genius_lyrics(self, track) -> str | None:
        artist = getattr(track, "grandparentTitle", None) or ""
        try:
            song = self.genius.search_song(track.title, artist)
        except Exception:
            return None
        if song is None or not song.lyrics or not song.lyrics.strip():
            return None
        return song.lyrics
