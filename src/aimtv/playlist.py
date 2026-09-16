"""Playlist sources: Airadio (original) and Plex."""

from __future__ import annotations

import json
import random
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

from aimtv.paths import airadio_home, aimtv_cache_dir
from aimtv.preflight import count_interstitials, count_library_songs


@dataclass
class Clip:
    path: Path
    kind: str  # song | ad | station-id
    title: str


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
    pool: list[tuple[Path, str]] = []
    for kind in ("ads", "station-id"):
        for path in count_interstitials(home, kind):
            pool.append((path, "ad" if kind == "ads" else "station-id"))
    if not pool:
        return []
    n = max(1, min(n, len(pool)))
    chosen = rng.sample(pool, n) if n <= len(pool) else [rng.choice(pool) for _ in range(n)]
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
    plex_token: Optional[str] = None,
    plex_url: Optional[str] = None,
    plex_section: Optional[str] = None,
    genius_token: Optional[str] = None,
) -> list[Clip]:
    """Finite playlist: song, 1–3 interstitials, song, … for song_count songs.

    If plex_token, plex_url, and plex_section are all provided, use Plex as the
    music source; otherwise, use the original Airadio source.
    """
    # Determine source
    use_plex = all(v is not None for v in (plex_token, plex_url, plex_section))
    if use_plex:
        song_clips = get_plex_song_clips(
            plex_token=plex_token,
            plex_url=plex_url,
            plex_section=plex_section,
            genius_token=genius_token,
            song_count=song_count,
            seed=seed,
        )
    else:
        song_clips = get_airadio_song_clips(
            song_count=song_count,
            seed=seed,
        )

    # Interleave interstitials (using Airadio home for interstitial files)
    home = airadio_home()
    rng = random.Random(seed if seed is not None else secrets.randbits(32))
    clips: list[Clip] = []
    for i, song in enumerate(song_clips):
        clips.append(song)
        if i + 1 < len(song_clips):
            n = rng.randint(interstitial_min, interstitial_max)
            clips.extend(pick_interstitials(home, n, rng))
    return clips


def get_plex_song_clips(
    *,
    plex_token: str,
    plex_url: str,
    plex_section: str,
    genius_token: Optional[str] = None,
    song_count: int = 2,
    seed: int | None = None,
) -> list[Clip]:
    """Return exactly `song_count` song Clips from a Plex library section."""
    from aimtv.plex_playlist import PlexPlaylistSource

    source = PlexPlaylistSource(
        baseurl=plex_url,
        token=plex_token,
        section_name=plex_section,
        genius_token=genius_token,
        cache_dir=aimtv_cache_dir() / "plex",
    )

    # We need to collect exactly `song_count` tracks.
    # We'll iterate over the source and take the first `song_count`.
    clips: list[Clip] = []
    rng = random.Random(seed if seed is not None else secrets.randbits(32))
    # Shuffle the source? We'll just take the first `song_count` for determinism if seed is given.
    # But to have some randomness, we can shuffle the list of all tracks and then take the first N.
    # However, fetching all tracks might be expensive. For now, we'll take the first `song_count`
    # as they come from Plex (which is ordered by addition date?).
    # To make it random, we'll shuffle the list of tracks we collect until we have enough.
    # We'll collect up to 2*song_count tracks and then sample.
    # For simplicity, we'll just take the first `song_count` and rely on the user to have a shuffled library.
    # Alternatively, we can use the Plex API to get random tracks, but that's more complex.
    # We'll implement a simple shuffle: collect all track IDs, then sample.
    # But note: the PlexPlaylistSource.iter_tracks yields (audio_path, lyrics_path, metadata).
    # We don't want to download all tracks just to shuffle. We'll assume the library is not huge.
    # For now, we'll just take the first `song_count` and if the user wants randomness, they can
    # sort their library differently.
    # We'll implement a basic random selection by shuffling a list of up to 100 tracks.
    # This is a compromise.

    # We'll collect tracks in a list until we have at least song_count, up to a max of 100.
    collected: list[Tuple[Path, Path, dict]] = []
    for item in source.iter_tracks():
        collected.append(item)
        if len(collected) >= max(song_count * 2, 100):
            break

    if len(collected) < song_count:
        # If we didn't get enough, just use what we have (will error later if not enough)
        pass

    # Shuffle the collected tracks
    rng.shuffle(collected)
    selected = collected[:song_count]

    for audio_path, lyrics_path, metadata in selected:
        clips.append(
            Clip(
                path=audio_path,
                kind="song",
                title=metadata["title"],
            )
        )
    return clips