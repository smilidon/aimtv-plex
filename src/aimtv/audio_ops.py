"""Audio helpers: duration, airadio-exact stitch, FFT feature extraction.

Playlist audio must match airadio's ffmpeg pipeline byte-for-byte for the same
clip sequence: tail_fade on songs/fill interstitials; acrossfade only for the
bridge interstitial → next song. No numpy every-join crossfade.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

# Copied exactly from airadio.radio — do not drift.
TAIL_FADE_MAX = 2.0
INTERSTITIAL_FADE_MAX = 2.0
CROSSFADE_S = 2.5


class ClipLike(Protocol):
    path: Path
    kind: str  # song | ad | station-id | …


@dataclass
class Features:
    rms: float
    bass: float
    mid: float
    treble: float
    beat: bool


def duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def _run_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def tail_fade(path: Path, out: Path, fade_max: float = TAIL_FADE_MAX) -> Path:
    """Identical to airadio.radio.tail_fade."""
    dur = duration(path)
    fade = min(fade_max, max(0.35, dur * 0.12))
    start = max(0.0, dur - fade)
    _run_cmd(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-af",
            f"afade=t=out:st={start:.3f}:d={fade:.3f}",
            str(out),
        ]
    )
    return out


def crossfade(a: Path, b: Path, out: Path, fade_s: float = CROSSFADE_S) -> Path:
    """Identical to airadio.radio.crossfade."""
    fade = min(fade_s, max(0.05, duration(a) - 0.05), max(0.05, duration(b) - 0.05))
    _run_cmd(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(a),
            "-i",
            str(b),
            "-filter_complex",
            f"acrossfade=d={fade:.3f}:c1=tri:c2=tri",
            str(out),
        ]
    )
    return out


def _hard_concat_wavs(paths: list[Path], out: Path) -> Path:
    """Sequential join with stream copy (same as consecutive airadio play() calls)."""
    if not paths:
        raise RuntimeError("no audio pieces to concatenate")
    if len(paths) == 1:
        out.parent.mkdir(parents=True, exist_ok=True)
        if paths[0].resolve() != out.resolve():
            out.write_bytes(paths[0].read_bytes())
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.parent / f"{out.stem}-concat.txt"
    lines: list[str] = []
    for path in paths:
        esc = str(path.resolve()).replace("'", r"'\''")
        lines.append(f"file '{esc}'")
    lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        _run_cmd(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(lst),
                "-c",
                "copy",
                str(out),
            ]
        )
    except subprocess.CalledProcessError:
        # Fallback if codecs/layouts differ between pieces.
        _run_cmd(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(lst),
                "-c:a",
                "pcm_s16le",
                str(out),
            ]
        )
    return out


def stitch_like_airadio(
    clips: list[ClipLike],
    work_dir: Path,
    out: Path,
) -> tuple[Path, list[float], list[float]]:
    """Build one WAV matching airadio's play rules for a finite playlist.

    - Songs (standalone): tail_fade(TAIL_FADE_MAX)
    - Fill interstitials (all but last before a song): tail_fade(INTERSTITIAL_FADE_MAX), hard join
    - Last interstitial before a song: raw + acrossfade into tail_faded song (bridge)
    - Hard concat between pieces (no every-join crossfade)

    Returns (out_wav, clip_starts, clip_durs) for title overlay timing.
    """
    if not clips:
        raise RuntimeError("empty playlist")

    work_dir.mkdir(parents=True, exist_ok=True)
    n = len(clips)
    starts = [0.0] * n
    durs = [duration(c.path) for c in clips]
    pieces: list[Path] = []
    t = 0.0
    i = 0
    piece_i = 0

    while i < n:
        clip = clips[i]
        if clip.kind == "song":
            piece = tail_fade(
                clip.path, work_dir / f"piece-{piece_i:04d}-song.wav", TAIL_FADE_MAX
            )
            piece_i += 1
            pd = duration(piece)
            starts[i] = t
            durs[i] = pd
            pieces.append(piece)
            t += pd
            i += 1
            continue

        # Interstitial run until next song (or end).
        j = i
        while j < n and clips[j].kind != "song":
            j += 1
        inters = list(range(i, j))

        if j >= n:
            for idx in inters:
                piece = tail_fade(
                    clips[idx].path,
                    work_dir / f"piece-{piece_i:04d}-inter.wav",
                    INTERSTITIAL_FADE_MAX,
                )
                piece_i += 1
                pd = duration(piece)
                starts[idx] = t
                durs[idx] = pd
                pieces.append(piece)
                t += pd
            i = j
            continue

        fill_idxs = inters[:-1]
        bridge_idx = inters[-1]
        song_idx = j

        for idx in fill_idxs:
            piece = tail_fade(
                clips[idx].path,
                work_dir / f"piece-{piece_i:04d}-inter.wav",
                INTERSTITIAL_FADE_MAX,
            )
            piece_i += 1
            pd = duration(piece)
            starts[idx] = t
            durs[idx] = pd
            pieces.append(piece)
            t += pd

        song_faded = tail_fade(
            clips[song_idx].path,
            work_dir / f"piece-{piece_i:04d}-song-bridged.wav",
            TAIL_FADE_MAX,
        )
        piece_i += 1
        bridge_raw = clips[bridge_idx].path
        fade = min(
            CROSSFADE_S,
            max(0.05, duration(bridge_raw) - 0.05),
            max(0.05, duration(song_faded) - 0.05),
        )
        bridged = crossfade(
            bridge_raw,
            song_faded,
            work_dir / f"piece-{piece_i:04d}-bridge.wav",
            CROSSFADE_S,
        )
        piece_i += 1

        dur_a = duration(bridge_raw)
        dur_b = duration(song_faded)
        starts[bridge_idx] = t
        durs[bridge_idx] = dur_a
        starts[song_idx] = t + max(0.0, dur_a - fade)
        durs[song_idx] = dur_b
        pieces.append(bridged)
        t += duration(bridged)
        i = song_idx + 1

    _hard_concat_wavs(pieces, out)
    return out, starts, durs


def load_mono(path: Path, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        n = int(len(mono) * target_sr / sr)
        x_old = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
        mono = np.interp(x_new, x_old, mono).astype(np.float32)
        sr = target_sr
    return mono, sr


def band_energy(mag: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean(np.square(mag[mask]))))


def features_at(
    mono: np.ndarray,
    sr: int,
    t: float,
    *,
    window_s: float = 0.05,
    prev_rms: float = 0.0,
    peak_rms: float = 0.05,
) -> Features:
    """Return band energies normalized roughly to 0–1 for visual mapping."""
    n = max(64, int(window_s * sr))
    center = int(t * sr)
    start = max(0, center - n // 2)
    end = min(len(mono), start + n)
    chunk = mono[start:end]
    if len(chunk) < 8:
        return Features(0.0, 0.0, 0.0, 0.0, False)
    if len(chunk) < n:
        chunk = np.pad(chunk, (0, n - len(chunk)))
    windowed = chunk * np.hanning(len(chunk))
    spec = np.fft.rfft(windowed)
    mag = np.abs(spec).astype(np.float32)
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sr)
    rms = float(np.sqrt(np.mean(np.square(chunk))))
    bass = band_energy(mag, freqs, 20, 200)
    mid = band_energy(mag, freqs, 200, 2000)
    treble = band_energy(mag, freqs, 2000, 12000)

    peak = max(float(peak_rms), rms, 1e-3)
    rms_n = float(np.clip(rms / peak, 0.0, 1.0))
    total = float(np.mean(mag)) + 1e-6
    bass_n = float(np.clip(bass / (total * 3.0), 0.0, 1.0))
    mid_n = float(np.clip(mid / (total * 3.0), 0.0, 1.0))
    treble_n = float(np.clip(treble / (total * 3.0), 0.0, 1.0))
    beat = rms_n > float(prev_rms) * 1.25 and rms_n > 0.08
    return Features(rms=rms_n, bass=bass_n, mid=mid_n, treble=treble_n, beat=beat)
