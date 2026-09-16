"""Startup checks: Airadio install, library assets, and GPU headroom."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from aimtv.paths import airadio_home

# Conservative free VRAM for SD-Turbo img2img @ ~512² + activations.
MIN_FREE_MIB = 6 * 1024


@dataclass
class GpuInfo:
    name: str
    total_mib: int
    free_mib: int
    util_pct: int
    processes: list[tuple[int, str, int]]  # pid, name, used_mib


def airadio_installed() -> bool:
    # AI MTV imports Airadio's provenance modules. A globally visible CLI is
    # not enough when AI MTV itself lives in an isolated pipx environment.
    return importlib.util.find_spec("airadio") is not None


def check_airadio_ready() -> int:
    """Explain how to satisfy AI MTV's Airadio prerequisites."""
    if not airadio_installed():
        print(
            "AI MTV needs Airadio before it can run. Install it with either:\n\n"
            "  pip install airadio\n"
            "  pipx install airadio\n\n"
            "Then run `airadio` for a bit to populate the music library, and "
            "start `aimtv` again.",
            flush=True,
        )
        return 1

    home = airadio_home()
    songs = count_library_songs(home)
    ads = count_interstitials(home, "ads")
    station = count_interstitials(home, "station-id")
    if len(songs) < 2 or len(ads) < 1 or len(station) < 1:
        print(
            "AI MTV found Airadio, but its music library is not populated enough yet.\n\n"
            f"  AIRADIO_HOME: {home}\n"
            f"  songs:        {len(songs)} (need at least 2)\n"
            f"  ads:          {len(ads)} (need at least 1)\n"
            f"  station IDs:  {len(station)} (need at least 1)\n\n"
            "Run `airadio` for a bit so it can populate the music library and "
            "interstitials, then start `aimtv` again.",
            flush=True,
        )
        return 1
    return 0


def count_library_songs(home: Path) -> list[Path]:
    library = home / "library"
    if not library.is_dir():
        return []
    quarantine = (library / "quarantine").resolve()
    songs = []
    for path in sorted(library.glob("*.wav")):
        try:
            path.resolve().relative_to(quarantine)
            continue
        except ValueError:
            pass
        songs.append(path)
    return songs


def count_interstitials(home: Path, kind: str) -> list[Path]:
    root = home / "interstitials" / "audio" / kind
    if not root.is_dir():
        return []
    return [p for p in root.rglob("*.wav") if "piper-backup" not in p.parts]


def query_gpu() -> GpuInfo | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        gpu_line = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()[0]
        name, total, free, util = [x.strip() for x in gpu_line.split(",")]
        procs: list[tuple[int, str, int]] = []
        try:
            proc_out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            for line in proc_out.splitlines():
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    procs.append((int(parts[0]), parts[1], int(float(parts[2]))))
        except subprocess.CalledProcessError:
            pass
        return GpuInfo(
            name=name,
            total_mib=int(float(total)),
            free_mib=int(float(free)),
            util_pct=int(float(util)),
            processes=procs,
        )
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return None


def preflight(*, min_free_mib: int = MIN_FREE_MIB) -> int:
    """Return 0 if OK, else print advice and return 1. Never kills processes."""
    if check_airadio_ready() != 0:
        return 1

    home = airadio_home()
    songs = count_library_songs(home)
    ads = count_interstitials(home, "ads")
    station = count_interstitials(home, "station-id")

    gpu = query_gpu()
    if gpu is None:
        print(
            "AI MTV needs an NVIDIA GPU with nvidia-smi available.\n"
            "No usable GPU was detected.",
            flush=True,
        )
        return 1

    if gpu.free_mib < min_free_mib:
        print(
            "GPU does not have enough free memory for AI MTV yet.\n\n"
            f"  GPU:   {gpu.name}\n"
            f"  Free:  {gpu.free_mib} MiB / {gpu.total_mib} MiB "
            f"(need ≥ {min_free_mib} MiB free)\n"
            f"  Util:  {gpu.util_pct}%\n",
            flush=True,
        )
        if gpu.processes:
            print("Processes currently using the GPU:", flush=True)
            for pid, name, used in gpu.processes:
                print(f"  pid {pid:>7}  {used:>6} MiB  {name}", flush=True)
            print(flush=True)
        print(
            "Please stop GPU-heavy apps yourself (for example Airadio / ComfyUI),\n"
            "then run `aimtv` again. AI MTV will not kill other processes.",
            flush=True,
        )
        return 1

    print(
        f"AI MTV preflight ok — {len(songs)} songs, {len(ads)} ads, "
        f"{len(station)} station-ids; GPU free {gpu.free_mib} MiB",
        flush=True,
    )
    return 0
