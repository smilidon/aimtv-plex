"""Load editable channel prompts from JSON (user file preferred)."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

from aimtv.paths import aimtv_home, ensure_aimtv_layout


@dataclass(frozen=True)
class PromptBank:
    prompts: tuple[str, ...]
    subjects: tuple[str, ...]
    style_suffix: str
    source: Path


def pick_random_prompt(rng: random.Random, bank: PromptBank) -> tuple[str, str]:
    """Compose one legacy random-channel prompt from the editable prompt bank."""
    subject = rng.choice(bank.subjects)
    channel = rng.choice(bank.prompts)
    bits = [channel, f"featuring {subject}"]
    if bank.style_suffix:
        bits.append(bank.style_suffix)
    else:
        bits.append(
            "highly detailed subject, sharp focus, vivid color, not blank, not empty, not abstract fog"
        )
    return ", ".join(bits), subject


def package_prompts_path() -> Path:
    """Path to the bundled default prompts.json."""
    return Path(str(resources.files("aimtv").joinpath("data/prompts.json")))


def user_prompts_path(home: Path | None = None) -> Path:
    return (home or aimtv_home()) / "prompts.json"


def ensure_user_prompts(home: Path | None = None) -> Path:
    """Copy packaged defaults into AIMTV_HOME if the user file is missing."""
    root = ensure_aimtv_layout(home)
    dest = user_prompts_path(root)
    if not dest.is_file():
        src = package_prompts_path()
        shutil.copy2(src, dest)
    return dest


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
    prompts = data.get("prompts") or []
    subjects = data.get("subjects") or []
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{path}: 'prompts' must be a non-empty list")
    if not isinstance(subjects, list) or not subjects:
        # Fall back to first few prompts as subjects.
        subjects = [str(p) for p in prompts[:8]]
    style = str(data.get("style_suffix") or "").strip()
    return {
        "prompts": tuple(str(p).strip() for p in prompts if str(p).strip()),
        "subjects": tuple(str(s).strip() for s in subjects if str(s).strip()),
        "style_suffix": style,
    }


@lru_cache(maxsize=4)
def load_prompt_bank(path: str | None = None) -> PromptBank:
    """
    Load prompts.

    Precedence:
      1. explicit path
      2. ~/.local/share/aimtv/prompts.json (created from package default if missing)
      3. packaged aimtv/data/prompts.json
    """
    if path:
        p = Path(path).expanduser().resolve()
        raw = _load_json(p)
        return PromptBank(source=p, **raw)

    user = ensure_user_prompts()
    try:
        raw = _load_json(user)
        return PromptBank(source=user, **raw)
    except (OSError, ValueError, json.JSONDecodeError):
        pkg = package_prompts_path()
        raw = _load_json(pkg)
        return PromptBank(source=pkg, **raw)


def reload_prompt_bank(path: str | None = None) -> PromptBank:
    load_prompt_bank.cache_clear()
    return load_prompt_bank(path)
