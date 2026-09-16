"""Consent-gated local model setup with progress UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from aimtv.paths import ensure_aimtv_layout, models_dir

# Compact turbo stack for PoC feedback img2img.
MODEL_ID = "stabilityai/sd-turbo"
MODEL_MARKER = "model_index.json"
# Rough Hub footprint for sd-turbo (fp16 weights dominate).
APPROX_SIZE_GB = 6.5
TOTAL_APPROX_SIZE_GB = 7.0


@dataclass
class VisionBundle:
    pipe: object
    device: str


def model_root() -> Path:
    ensure_aimtv_layout()
    return models_dir() / "sd-turbo"


def models_present(root: Path | None = None) -> bool:
    path = root or model_root()
    # Require index + unet weights so mid-download never looks "ready".
    return (path / MODEL_MARKER).is_file() and any(path.joinpath("unet").glob("*.safetensors"))


def _ask_permission() -> bool:
    print(
        "\n"
        "AI MTV needs diffusion weights before it can paint TV frames.\n"
        f"  Model:     {MODEL_ID}\n"
        f"  Dest:      {model_root()}\n"
        f"  Size:      ~{APPROX_SIZE_GB:.1f} GB (approximate)\n"
        "  Warning:   this takes time and uses substantial disk space.\n",
        flush=True,
    )
    try:
        answer = input("Allow AI MTV to download this model now? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    return answer in {"y", "yes"}


def fetch_models(*, assume_yes: bool = False) -> Path:
    dest = model_root()
    dest.mkdir(parents=True, exist_ok=True)
    if models_present(dest):
        print(f"models already present at {dest}", flush=True)
        return dest

    if not assume_yes and not _ask_permission():
        print(
            "Download declined. Run `aimtv fetch-models` when you are ready.",
            flush=True,
        )
        raise SystemExit(1)

    print(f"downloading {MODEL_ID} → {dest}", flush=True)
    # huggingface_hub shows tqdm progress bars by default in a TTY / when enabled.
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(dest),
        tqdm_class=tqdm,
    )
    if not models_present(dest):
        raise RuntimeError(f"download finished but {MODEL_MARKER} missing under {dest}")
    print("download complete.", flush=True)
    return dest


def required_models_present() -> bool:
    from aimtv.alignment import alignment_model_present

    return models_present() and alignment_model_present()


def _ask_required_models_permission(*, vision_missing: bool, voice_missing: bool) -> bool:
    from aimtv.alignment import ASR_APPROX_GB, ASR_MODEL_ID, alignment_model_root

    print(
        "\nAI MTV needs local model files before it can generate videos.\n"
        "Nothing is sent to a hosted inference service.\n",
        flush=True,
    )
    if voice_missing:
        print(
            f"  Voice timing: {ASR_MODEL_ID}\n"
            f"  Destination:  {alignment_model_root()}\n"
            f"  Size:         ~{ASR_APPROX_GB:.1f} GB\n",
            flush=True,
        )
    if vision_missing:
        print(
            f"  Video model:  {MODEL_ID}\n"
            f"  Destination:  {model_root()}\n"
            f"  Size:         ~{APPROX_SIZE_GB:.1f} GB\n",
            flush=True,
        )
    print(
        f"  Maximum first-run download: ~{TOTAL_APPROX_SIZE_GB:.1f} GB\n"
        "  Warning: this can take time and uses substantial disk space.\n",
        flush=True,
    )
    try:
        answer = input("Allow AI MTV to download the missing models now? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    return answer in {"y", "yes"}


def fetch_required_models(*, assume_yes: bool = False) -> tuple[Path, Path]:
    """Install every inference model AI MTV requires, with one consent prompt."""
    from aimtv.alignment import (
        alignment_model_present,
        alignment_model_root,
        fetch_alignment_model,
    )

    voice_missing = not alignment_model_present()
    vision_missing = not models_present()
    if not voice_missing and not vision_missing:
        print("AI MTV models are already installed.", flush=True)
        return alignment_model_root(), model_root()

    if not assume_yes and not _ask_required_models_permission(
        vision_missing=vision_missing,
        voice_missing=voice_missing,
    ):
        print(
            "Model download declined. Run `aimtv fetch-models` when you are ready.",
            flush=True,
        )
        raise SystemExit(1)

    voice = (
        fetch_alignment_model(assume_yes=True)
        if voice_missing
        else alignment_model_root()
    )
    vision = fetch_models(assume_yes=True) if vision_missing else model_root()
    print("AI MTV model setup complete.", flush=True)
    return voice, vision


def load_pipeline(device: str | None = None) -> VisionBundle:
    import torch
    from diffusers import AutoPipelineForImage2Image

    dest = model_root()
    if not models_present(dest):
        raise RuntimeError(f"models missing at {dest}; fetch first")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"loading {MODEL_ID} on {device}…", flush=True)
    pipe = AutoPipelineForImage2Image.from_pretrained(
        str(dest),
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    )
    pipe = pipe.to(device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    print("model loaded.", flush=True)
    return VisionBundle(pipe=pipe, device=device)
