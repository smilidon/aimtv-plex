"""Library-only playlist: songs and interstitials from AIRADIO_HOME."""

from __future__ import annotations

import json
import random
import secrets
from dataclasses import dataclass
from pathlib import Path

from aimtv.paths import airadio_home
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


def build_review_playlist(
    *,
    song_count: int = 2,
    interstitial_min: int = 1,
    interstitial_max: int = 3,
    seed: int | None = None,
) -> list[Clip]:
    """Finite playlist: song, 1–3 interstitials, song, … for song_count songs."""
    home = airadio_home()
    songs = count_library_songs(home)
    if len(songs) < song_count:
        raise RuntimeError(f"need ≥ {song_count} library songs, found {len(songs)}")
    rng = random.Random(seed if seed is not None else secrets.randbits(32))
    titles = _catalog_titles(home)
    picks = rng.sample(songs, song_count)
    clips: list[Clip] = []
    for i, song in enumerate(picks):
        clips.append(
            Clip(path=song, kind="song", title=title_for(song, home, titles))
        )
        if i + 1 < song_count:
            n = rng.randint(interstitial_min, interstitial_max)
            clips.extend(pick_interstitials(home, n, rng))
    return clips
