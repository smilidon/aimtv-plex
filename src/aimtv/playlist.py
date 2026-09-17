"""Playlist sources: Airadio (original) and Plex."""

from __future__ import annotations

import json
import random
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from aimtv.paths import airadio_home, aimtv_cache_dir
from aimtv.preflight import count_interstitials, count_library_songs


@dataclass
class Clip:
    path: Path
    kind: str  # song | ad | station-id
    title: str
    # Set only for songs that carry their own lyrics (e.g. Plex tracks). Airadio
    # clips leave these unset and are resolved through the catalog/provenance.
    lyrics_path: Path | None = None
    lyrics_source: str | None = None
    provenance_id: str | None = None


@dataclass(frozen=True)
class PlexConfig:
    """Connection details for sourcing songs from a Plex music section."""

    url: str
    token: str = field(repr=False)
    section: str
    genius_token: str | None = field(default=None, repr=False)


def _catalog_titles(home: Path) -> dict[str, str]:
    catalog = home / "library" / "catalog.json"
    if not catalog.is_file():
        return {}
    try:
        data = json.loads(catalog.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out: dict[str, str] = {}
    for entry in data.get("songs", []):
        rel = entry.get("path")
        title = entry.get("title")
        if rel and title:
            out[str((home / rel).resolve())] = str(title)
    return out


def title_for(path: Path, home: Path, titles: dict[str, str]) -> str:
    key = str(path.resolve())
    if key in titles:
        return titles[key]
    return path.stem.replace("-", " ").title()


def pick_interstitials(home: Path, n: int, rng: random.Random) -> list[Clip]:
    if n <= 0:
        return []
    pool: list[tuple[Path, str]] = []
    for kind in ("ads", "station-id"):
        for path in sorted(count_interstitials(home, kind)):
            pool.append((path, "ad" if kind == "ads" else "station-id"))
    if not pool:
        return []
    n = min(n, len(pool))
    chosen = rng.sample(pool, n)
    return [Clip(path=p, kind=k, title=p.stem) for p, k in chosen]


def get_airadio_song_clips(
    *,
    song_count: int = 2,
    seed: int | None = None,
) -> list[Clip]:
    """Return exactly `song_count` song Clips from the Airadio library."""
    home = airadio_home()
    songs = count_library_songs(home)
    if len(songs) < song_count:
        raise RuntimeError(f"need ≥ {song_count} library songs, found {len(songs)}")
    rng = random.Random(seed if seed is not None else secrets.randbits(32))
    titles = _catalog_titles(home)
    picks = rng.sample(songs, song_count)
    clips: list[Clip] = []
    for song in picks:
        clips.append(
            Clip(path=song, kind="song", title=title_for(song, home, titles))
        )
    return clips


def build_review_playlist(
    *,
    song_count: int = 2,
    interstitial_min: int = 1,
    interstitial_max: int = 3,
    seed: int | None = None,
    plex: PlexConfig | None = None,
) -> list[Clip]:
    """Finite playlist: song, 1–3 interstitials, song, … for song_count songs.

    Songs come from Plex when `plex` is given, otherwise from the Airadio
    library. Interstitials always come from the Airadio library.
    """
    if plex is not None:
        song_clips = get_plex_song_clips(plex, song_count=song_count, seed=seed)
    else:
        song_clips = get_airadio_song_clips(song_count=song_count, seed=seed)

    home = airadio_home()
    rng = random.Random(seed if seed is not None else secrets.randbits(32))
    clips: list[Clip] = []
    for i, song in enumerate(song_clips):
        clips.append(song)
        if i + 1 < len(song_clips):
            n = rng.randint(interstitial_min, interstitial_max)
            clips.extend(pick_interstitials(home, n, rng))
    return clips


# Upper bound on tracks pulled from Plex before sampling, so a large library
# does not force a full scan (and a full lyric lookup) for a two-song review.
PLEX_SAMPLE_POOL = 100


def get_plex_song_clips(
    plex: PlexConfig,
    *,
    song_count: int = 2,
    seed: int | None = None,
) -> list[Clip]:
    """Return exactly `song_count` song Clips sampled from a Plex library section."""
    from aimtv.plex_playlist import PlexPlaylistSource

    source = PlexPlaylistSource(
        baseurl=plex.url,
        token=plex.token,
        section_name=plex.section,
        genius_token=plex.genius_token,
        cache_dir=aimtv_cache_dir() / "plex",
    )
    rng = random.Random(seed if seed is not None else secrets.randbits(32))

    # Sample rating keys first (cheap metadata), then resolve only the chosen
    # tracks so lyric extraction/lookup runs for song_count tracks, not the pool.
    pool = source.track_keys(limit=max(song_count, PLEX_SAMPLE_POOL))
    if len(pool) < song_count:
        raise RuntimeError(
            f"Plex section '{plex.section}' has {len(pool)} track(s); need at least {song_count}."
        )
    chosen = rng.sample(pool, song_count)

    clips: list[Clip] = []
    for audio_path, lyrics_path, metadata in source.resolve_tracks(chosen):
        clips.append(
            Clip(
                path=audio_path,
                kind="song",
                title=str(metadata["title"]),
                lyrics_path=lyrics_path,
                lyrics_source=metadata.get("lyrics_source"),
                provenance_id=(
                    f"plex:{metadata['server_id']}:{metadata['rating_key']}"
                    if metadata.get("server_id") else f"plex:{metadata['rating_key']}"
                ),
            )
        )
    return clips

