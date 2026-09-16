"""Plex playlist source for AIMTV."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Tuple, Optional

from plexapi.server import PlexServer
from plexapi.exceptions import NotFound, BadRequest
from mutagen.id3 import ID3, USLT, ID3NoHeaderError
from lyricsgenius import Genius

from aimtv.paths import aimtv_cache_dir


class PlexPlaylistSource:
    """
    A source of (audio_path, lyrics_path, metadata) for tracks in a Plex library section.

    Parameters
    ----------
    baseurl : str
        Base URL of the Plex server (e.g., "http://localhost:32400").
    token : str
        Plex token for authentication.
    section_name : str
        Name of the library section containing music (e.g., "Music").
    genius_token : Optional[str]
        Token for Genius API (for lyric fallback). If None, no online lookup is attempted.
    cache_dir : Optional[Path]
        Directory to cache downloaded audio and lyrics. If None, uses aimtv_cache_dir().
    """

    def __init__(
        self,
        baseurl: str,
        token: str,
        section_name: str,
        genius_token: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.plex = PlexServer(baseurl, token)
        try:
            self.section = self.plex.library.section(section_name)
        except (NotFound, BadRequest) as exc:
            raise ValueError(f"Plex section '{section_name}' not found or inaccessible.") from exc

        self.genre = Genius(genius_token) if genius_token else None
        self.cache_dir = cache_dir or aimtv_cache_dir() / "plex"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def iter_tracks(self) -> Iterable[Tuple[Path, Path, dict]]:
        """
        Yield tuples of (audio_path, lyrics_path, metadata) for each track in the section.

        Metadata is a dict with at least:
            - title: str
            - artist: str
            - album: Optional[str]
        """
        for track in self.section.tracks():
            # Try to get the local file path first
            audio_path = self._get_local_file_path(track)
            if audio_path is None:
                # Fallback: stream to a temporary file
                audio_path = self._stream_to_temp(track)

            # Extract or fetch lyrics
            lyrics_path = self._get_lyrics_path(track, audio_path)

            # Build metadata
            metadata = {
                "title": track.title,
                "artist": track.artist().title if hasattr(track.artist(), 'title') else str(track.artist()),
                "album": track.album().title if hasattr(track.album(), 'title') else None,
            }

            yield audio_path, lyrics_path, metadata

    def _get_local_file_path(self, track) -> Optional[Path]:
        """
        Return the local file path if the track is accessible directly, else None.
        """
        if hasattr(track, "locations") and track.locations:
            # Plex may return multiple locations; take the first one that exists
            for loc in track.locations:
                p = Path(loc)
                if p.exists():
                    return p
        return None

    def _stream_to_temp(self, track) -> Path:
        """
        Download the media file to a temporary location in the cache.
        Returns the path to the cached file.
        """
        # Use a hash of the track's ratingKey and version to name the cache file
        cache_key = f"{track.ratingKey}_{getattr(track, 'version', '')}"
        cache_path = self.cache_dir / f"{cache_key}.tmp"

        # If we already have a cached file, return it
        if cache_path.exists():
            return cache_path

        # Otherwise, download the media
        try:
            media = track.media[0]
            part = media.parts[0]
            # Note: This requires the Plex server to allow direct access (or token)
            download_url = part.getStreamURL(**{'X-Plex-Token': self.plex._token})
            # We'll use requests to download, but for simplicity we assume the user can
            # mount the media directory or use Plex's local file access.
            # For now, we'll just note that this is a placeholder and in practice
            # the user should ensure the media is accessible via local file paths.
            # In a real implementation, we would stream the file here.
            raise NotImplementedError(
                "Direct streaming from Plex is not implemented. "
                "Please ensure your Plex server hosts the media files on a local filesystem "
                "accessible to this process, or implement a download function."
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to stream track '{track.title}' from Plex.") from exc

    def _get_lyrics_path(self, track, audio_path: Path) -> Path:
        """
        Return a Path to a lyrics file (.txt) for the given track.
        First tries embedded lyrics (ID3 USLT), then online lookup via Genius.
        Cached in the cache directory.
        """
        # Try to extract embedded lyrics
        embedded_lyrics = self._extract_embedded_lyrics(audio_path)
        if embedded_lyrics is not None:
            # Cache the embedded lyrics
            lyric_cache = self.cache_dir / f"{audio_path.stem}_lyrics.txt"
            if not lyric_cache.exists():
                lyric_cache.write_text(embedded_lyrics, encoding="utf-8")
            return lyric_cache

        # Fallback to online lookup
        if self.genre is None:
            # No online lookup available; return an empty lyrics file
            lyric_cache = self.cache_dir / f"{audio_path.stem}_lyrics.txt"
            if not lyric_cache.exists():
                lyric_cache.write_text("", encoding="utf-8")
            return lyric_cache

        # Use Genius to search for the song
        try:
            song = self.genre.search_song(track.title, track.artist().title if hasattr(track.artist(), 'title') else str(track.artist()))
            if song is not None and song.lyrics:
                lyric_cache = self.cache_dir / f"{audio_path.stem}_lyrics.txt"
                if not lyric_cache.exists():
                    lyric_cache.write_text(song.lyrics, encoding="utf-8")
                return lyric_cache
        except Exception:
            pass  # Fall back to empty lyrics

        # If all else fails, return an empty lyrics file
        lyric_cache = self.cache_dir / f"{audio_path.stem}_lyrics.txt"
        if not lyric_cache.exists():
            lyric_cache.write_text("", encoding="utf-8")
        return lyric_cache

    def _extract_embedded_lyrics(self, audio_path: Path) -> Optional[str]:
        """
        Attempt to extract lyrics from ID3 USLT tag.
        Returns the lyrics string if found, else None.
        """
        try:
            audio_file = ID3(audio_path)
            for tag in audio_file.values():
                if isinstance(tag, USLT):
                    return tag.text
        except ID3NoHeaderError:
            pass
        except Exception:
            pass
        return None