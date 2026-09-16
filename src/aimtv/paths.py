from __future__ import annotations

import os
from pathlib import Path


def aimtv_home() -> Path:
    raw = os.environ.get("AIMTV_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".local/share/aimtv").resolve()


def models_dir(home: Path | None = None) -> Path:
    return (home or aimtv_home()) / "models"


def output_dir(home: Path | None = None) -> Path:
    return (home or aimtv_home()) / "output"


def alignments_dir(home: Path | None = None) -> Path:
    return (home or aimtv_home()) / "alignments"


def airadio_home() -> Path:
    raw = os.environ.get("AIRADIO_HOME") or os.environ.get("IDR_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    default = Path.home() / ".local/share/airadio"
    cwd_lib = Path.cwd() / "library"
    if cwd_lib.is_dir():
        return Path.cwd().resolve()
    return default.resolve()


def ensure_aimtv_layout(home: Path | None = None) -> Path:
    root = home or aimtv_home()
    models_dir(root).mkdir(parents=True, exist_ok=True)
    output_dir(root).mkdir(parents=True, exist_ok=True)
    alignments_dir(root).mkdir(parents=True, exist_ok=True)
    # Seed editable prompts.json once so users can tweak without hunting package files.
    prompts = root / "prompts.json"
    if not prompts.is_file():
        try:
            from importlib import resources
            import shutil

            src = resources.files("aimtv").joinpath("data/prompts.json")
            shutil.copy2(str(src), prompts)
        except Exception:
            pass
    return root
