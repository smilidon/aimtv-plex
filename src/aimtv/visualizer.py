"""Reactive diffusion visualizer — high-mutation feedback img2img."""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from aimtv.audio_ops import Features
from aimtv.models import VisionBundle
from aimtv.prompts import PromptBank, load_prompt_bank, pick_random_prompt

# Even with silence, mutate hard every frame (do not preserve prior blank).
BASE_STRENGTH = 0.78
MAX_STRENGTH = 0.98
STEPS = 4
RESEED_EVERY_S = 3.5
COLLAPSE_FRAMES = 3
VAR_MIN = 18.0
EDGE_MIN = 0.012


@dataclass
class FrameState:
    image: Image.Image
    prompt: str
    channel_t: float
    prev_rms: float = 0.0
    low_detail_streak: int = 0
    frames_since_reseed: float = 0.0
    subject: str = ""


def _bank() -> PromptBank:
    return load_prompt_bank()


def _pick_prompt(rng: random.Random, bank: PromptBank | None = None) -> tuple[str, str]:
    return pick_random_prompt(rng, bank or _bank())


def _to_pil(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _to_bgr(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _fresh_noise(width: int, height: int, rng: random.Random) -> Image.Image:
    gen = np.random.default_rng(rng.randint(0, 2**31 - 1))
    # Structured noise (not flat gray) so img2img has edges to latch onto.
    base = gen.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    blobs = gen.integers(0, 256, size=(max(8, height // 16), max(8, width // 16), 3), dtype=np.uint8)
    blobs = cv2.resize(blobs, (width, height), interpolation=cv2.INTER_CUBIC)
    mixed = cv2.addWeighted(base, 0.45, blobs, 0.55, 0)
    mixed = cv2.GaussianBlur(mixed, (0, 0), 1.2)
    return _to_pil(mixed)


def init_frame(width: int, height: int, rng: random.Random) -> FrameState:
    prompt, subject = _pick_prompt(rng)
    return FrameState(
        image=_fresh_noise(width, height, rng),
        prompt=prompt,
        channel_t=0.0,
        subject=subject,
    )


def frame_detail_stats(image: Image.Image) -> tuple[float, float]:
    gray = cv2.cvtColor(_to_bgr(image), cv2.COLOR_BGR2GRAY)
    variance = float(np.var(gray))
    edges = cv2.Canny(gray, 60, 140)
    edge_density = float(np.mean(edges > 0))
    return variance, edge_density


def maybe_surf_channel(state: FrameState, dt: float, rng: random.Random) -> FrameState:
    state.channel_t += dt
    interval = rng.uniform(18.0, 55.0)
    if state.channel_t < interval:
        return state
    state.channel_t = 0.0
    state.prompt, state.subject = _pick_prompt(rng)
    # Hard CRT reset into fresh noise so the new channel starts inventing again.
    w, h = state.image.size
    state.image = _fresh_noise(w, h, rng)
    state.frames_since_reseed = 0.0
    state.low_detail_streak = 0
    return state


def warp_frame(image: Image.Image, feat: Features, noise_amp: float) -> Image.Image:
    arr = _to_bgr(image)
    h, w = arr.shape[:2]
    # Stronger geometric motion so silence still moves the canvas.
    zoom = 1.0 + 0.03 + float(feat.bass) * 0.10
    angle = 1.5 + float(feat.mid) * 8.0
    dx = (float(feat.mid) - 0.5) * 24.0
    dy = (float(feat.bass) - 0.5) * 12.0
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, zoom)
    M[0, 2] += dx
    M[1, 2] += dy
    warped = cv2.warpAffine(
        arr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    amp = int(np.clip(25 + noise_amp * 90, 25, 120))
    noise = np.random.randint(0, amp, size=warped.shape, dtype=np.uint8)
    warped = cv2.add(warped, noise)
    return _to_pil(warped)


def _txt2img_like(bundle: VisionBundle, prompt: str, size: tuple[int, int], rng: random.Random) -> Image.Image:
    """Force a brand-new invented frame via near-total noise rewrite."""
    w, h = size
    noise = _fresh_noise(w, h, rng)
    generator = None
    try:
        import torch

        if bundle.device == "cuda":
            generator = torch.Generator(device="cuda").manual_seed(rng.randint(0, 2**31 - 1))
    except Exception:
        generator = None
    result = bundle.pipe(
        prompt=prompt,
        image=noise,
        num_inference_steps=STEPS,
        guidance_scale=0.0,
        strength=1.0,
        generator=generator,
    ).images[0]
    if result.size != (w, h):
        result = result.resize((w, h), Image.Resampling.LANCZOS)
    return result


def step_frame(
    bundle: VisionBundle,
    state: FrameState,
    feat: Features,
    *,
    rng: random.Random,
    dt: float,
    title: str | None = None,
    prompt_override: str | None = None,
    prompt_changed: bool = False,
    hard_reset: bool = False,
) -> FrameState:
    # Features are expected already normalized to ~0–1.
    rms = float(np.clip(feat.rms, 0.0, 1.0))
    bass = float(np.clip(feat.bass, 0.0, 1.0))
    mid = float(np.clip(feat.mid, 0.0, 1.0))
    treble = float(np.clip(feat.treble, 0.0, 1.0))
    beat = 1.0 if feat.beat else 0.0

    if prompt_override is None:
        state = maybe_surf_channel(state, dt, rng)
    elif prompt_changed:
        state.prompt = prompt_override
        state.subject = ""
        state.channel_t = 0.0
    state.frames_since_reseed += dt

    # Periodic hard reset — do not treat previous frame as sacred.
    need_reseed = (
        hard_reset
        or state.frames_since_reseed >= RESEED_EVERY_S
        or beat > 0 and rng.random() < 0.35
    )

    variance, edge_density = frame_detail_stats(state.image)
    if variance < VAR_MIN or edge_density < EDGE_MIN:
        state.low_detail_streak += 1
    else:
        state.low_detail_streak = 0
    if state.low_detail_streak >= COLLAPSE_FRAMES:
        need_reseed = True

    prompt = prompt_override or state.prompt
    # Legacy random channels have a separate subject to emphasize. Contextual
    # prompts are already ranked phrases and must not be rewritten here.
    if state.subject:
        weight = 1.0 + mid * 0.8 + treble * 0.5
        prompt = f"({state.subject}:{weight:.2f}), {prompt}"
    if title:
        prompt = f"{prompt}, mood from the song titled {title}"

    if need_reseed:
        state.image = _txt2img_like(bundle, prompt, state.image.size, rng)
        state.frames_since_reseed = 0.0
        state.low_detail_streak = 0
        state.prev_rms = rms
        return state

    noise_amp = 0.35 + treble * 0.65 + beat * 0.25
    warped = warp_frame(state.image, feat, noise_amp=noise_amp)

    # High baseline strength so silence still invents; audio pushes harder.
    denoise = BASE_STRENGTH + rms * 0.14 + bass * 0.04 + beat * 0.06
    denoise = float(np.clip(denoise, BASE_STRENGTH, MAX_STRENGTH))

    generator = None
    try:
        import torch

        if bundle.device == "cuda":
            # Seed perturbation from audio + beat punches.
            seed = rng.randint(0, 2**31 - 1) ^ int(bass * 1_000_003) ^ int(beat * 9_999_991)
            generator = torch.Generator(device="cuda").manual_seed(seed & 0x7FFFFFFF)
    except Exception:
        generator = None

    result = bundle.pipe(
        prompt=prompt,
        image=warped,
        num_inference_steps=STEPS,
        guidance_scale=0.0,
        strength=denoise,
        generator=generator,
    ).images[0]
    if result.size != warped.size:
        result = result.resize(warped.size, Image.Resampling.LANCZOS)

    # If the model still collapsed, immediately replace with a fresh invented frame.
    variance, edge_density = frame_detail_stats(result)
    if variance < VAR_MIN or edge_density < EDGE_MIN:
        result = _txt2img_like(bundle, prompt, warped.size, rng)
        state.frames_since_reseed = 0.0
        state.low_detail_streak = 0

    state.image = result
    state.prev_rms = rms
    return state
