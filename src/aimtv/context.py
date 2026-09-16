"""Resolve verified lyric context from an Airadio library or a Plex-sourced clip."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from airadio import interstitial_provenance, lyricist

from aimtv.playlist import Clip


@dataclass(frozen=True)
class ClipContext:
    audio: Path
    kind: str
    title: str
    lyrics: str | None
    lyrics_sha256: str | None
    provenance_id: str | None
    verified: bool
    source: str
    warning: str | None = None


def _song_catalog(home: Path) -> dict[str, dict]:
    path = home / "library" / "catalog.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str((home / entry["path"]).resolve()): entry
        for entry in data.get("songs", [])
        if isinstance(entry, dict) and entry.get("path")
    }


def _unverified(clip: Clip, source: str, warning: str) -> ClipContext:
    return ClipContext(
        audio=clip.path,
        kind=clip.kind,
        title=clip.title,
        lyrics=None,
        lyrics_sha256=None,
        provenance_id=None,
        verified=False,
        source=source,
        warning=warning,
    )


def _sourced_context(clip: Clip) -> ClipContext:
    """Context for a song that brought its own lyrics file (Plex tracks).

    The text is not reconstructed from a catalog, so traceability comes from
    recording where it came from (embedded tag or Genius) and its hash.
    """
    try:
        lyrics = Path(clip.lyrics_path).read_text(encoding="utf-8")
    except OSError as exc:
        return _unverified(clip, "plex", f"lyrics file unreadable: {exc}")
    if not lyrics.strip():
        return _unverified(clip, "plex", "no embedded lyrics and no Genius match")
    digest = hashlib.sha256(lyrics.encode("utf-8")).hexdigest()
    source = clip.lyrics_source or "unknown"
    return ClipContext(
        audio=clip.path,
        kind=clip.kind,
        title=clip.title,
        lyrics=lyrics,
        lyrics_sha256=digest,
        provenance_id=f"{clip.provenance_id or 'plex'}:{source}:{digest[:16]}",
        verified=True,
        source=f"plex {source} lyrics",
    )


def resolve_contexts(clips: list[Clip], airadio_home: Path) -> list[ClipContext]:
    """Resolve exact lyrics for every clip, never treating unverified text as fact."""
    home = interstitial_provenance.canonical_home(airadio_home)
    catalog = _song_catalog(home)
    contexts: list[ClipContext] = []
    for clip in clips:
        if clip.kind == "song" and clip.lyrics_path is not None:
            contexts.append(_sourced_context(clip))
            continue
        if clip.kind == "song":
            entry = catalog.get(str(clip.path.resolve()))
            if not entry:
                contexts.append(_unverified(clip, "song-catalog", "song catalog entry missing"))
                continue
            lyric_id = entry.get("lyric_id")
            expected_hash = entry.get("lyrics_sha256")
            title = str(entry.get("title") or clip.title)
            if not isinstance(lyric_id, int) or not expected_hash:
                contexts.append(
                    _unverified(clip, "song-catalog", "song lyric ID or hash missing")
                )
                continue
            lyrics = lyricist.compose_lyrics(title, lyric_id)
            actual_hash = lyricist.lyrics_sha256(lyrics)
            if actual_hash != expected_hash:
                contexts.append(
                    _unverified(clip, "song-catalog", "reconstructed song lyrics failed hash verification")
                )
                continue
            contexts.append(
                ClipContext(
                    audio=clip.path,
                    kind=clip.kind,
                    title=title,
                    lyrics=lyrics,
                    lyrics_sha256=actual_hash,
                    provenance_id=f"song-lyrics:{lyric_id}:{actual_hash[:16]}",
                    verified=True,
                    source="airadio deterministic lyric reservation",
                )
            )
            continue

        try:
            record = interstitial_provenance.load_record(clip.path)
            lyrics = interstitial_provenance.read_lyrics(clip.path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            contexts.append(_unverified(clip, "interstitial-provenance", str(exc)))
            continue
        contexts.append(
            ClipContext(
                audio=clip.path,
                kind=clip.kind,
                title=clip.title,
                lyrics=lyrics,
                lyrics_sha256=str(record["lyrics"]["sha256"]),
                provenance_id=str(record["generation_id"]),
                verified=True,
                source="airadio interstitial provenance",
            )
        )
    return contexts
