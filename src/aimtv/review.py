"""Review build: 2 library songs + interstitials → one MP4 with reactive video."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from aimtv.audio_ops import features_at, load_mono
from aimtv.alignment import ensure_alignments
from aimtv.models import VisionBundle, fetch_required_models, load_pipeline
from aimtv.paths import output_dir
from aimtv.planning import apply_alignments, build_review_plan, write_manifest
from aimtv.preflight import preflight
from aimtv.visualizer import init_frame, step_frame


def render_review_mp4(
    *,
    out: Path | None = None,
    song_count: int = 2,
    interstitial_min: int = 1,
    interstitial_max: int = 3,
    fps: float = 8.0,
    width: int = 512,
    height: int = 320,
    seed: int | None = None,
    assume_yes_models: bool = False,
    min_random_gap_s: float = 3.0,
    max_random_gap_s: float = 8.0,
    min_random_duration_s: float = 7.0,
    max_random_duration_s: float = 16.0,
    max_seconds: float | None = None,
    voice_alignment: bool = True,
) -> Path:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if width < 64 or height < 64 or width % 8 or height % 8:
        raise ValueError("width and height must be multiples of 8 and at least 64")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    code = preflight()
    if code != 0:
        raise SystemExit(code)

    plan = build_review_plan(
        song_count=song_count,
        interstitial_min=interstitial_min,
        interstitial_max=interstitial_max,
        seed=seed,
        min_random_gap_s=min_random_gap_s,
        max_random_gap_s=max_random_gap_s,
        min_random_duration_s=min_random_duration_s,
        max_random_duration_s=max_random_duration_s,
    )
    print(f"review seed: {plan.seed}", flush=True)
    print("review playlist:", flush=True)
    for clip, context in zip(plan.clips, plan.contexts):
        status = "verified" if context.verified else f"UNVERIFIED: {context.warning}"
        print(f"  [{clip.kind}] {context.title} — {clip.path.name} ({status})", flush=True)
    print(f"render manifest: {plan.manifest_path}", flush=True)
    unverified = [context for context in plan.contexts if not context.verified]
    if unverified:
        details = "; ".join(
            f"{context.audio.name}: {context.warning}" for context in unverified
        )
        raise RuntimeError(f"refusing an untraceable contextual render: {details}")

    fetch_required_models(assume_yes=assume_yes_models)

    if voice_alignment:
        alignments = ensure_alignments(
            plan.clips,
            plan.contexts,
            assume_yes_models=assume_yes_models,
        )
        plan = apply_alignments(
            plan,
            alignments,
            min_random_gap_s=min_random_gap_s,
            max_random_gap_s=max_random_gap_s,
            min_random_duration_s=min_random_duration_s,
            max_random_duration_s=max_random_duration_s,
        )
        for clip, context in zip(plan.clips, plan.contexts):
            alignment = plan.alignments.get(str(clip.path.resolve()))
            if alignment is not None:
                anchors = sum(
                    1
                    for unit in alignment.units
                    if unit.confidence >= 0.30 and len(unit.matched_words) >= 2
                )
                print(
                    f"  timing [{clip.kind}] {context.title}: "
                    f"matched {alignment.matched_token_ratio:.0%}, anchors {anchors}",
                    flush=True,
                )

    bundle: VisionBundle = load_pipeline()

    staging = output_dir() / "review-staging"
    mono, sr = load_mono(plan.audio_wav)
    timeline_s = len(mono) / sr
    total_s = min(timeline_s, max_seconds) if max_seconds is not None else timeline_s
    if total_s <= 0:
        raise ValueError("render duration must be positive")
    n_frames = max(1, int(total_s * fps))
    dt = 1.0 / fps

    rng = random.Random(plan.seed)
    state = init_frame(width, height, rng)

    raw_video = staging / "frames.avi"
    writer = cv2.VideoWriter(
        str(raw_video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {raw_video}")

    prev_rms = 0.0
    peak_rms = float(np.sqrt(np.mean(np.square(mono)))) * 2.5 + 1e-3
    previous_event_id: str | None = None
    previous_event_type: str | None = None
    try:
        for i in tqdm(range(n_frames), desc="aimtv frames", unit="frm"):
            t = i * dt
            feat = features_at(mono, sr, t, prev_rms=prev_rms, peak_rms=peak_rms)
            prev_rms = feat.rms
            event = plan.schedule.event_at(t)
            changed = event.event_id != previous_event_id
            reset = changed and (
                event.hard_reset or previous_event_type == "random-cutaway"
            )
            state = step_frame(
                bundle,
                state,
                feat,
                rng=rng,
                dt=dt,
                prompt_override=event.prompt,
                prompt_changed=changed,
                hard_reset=reset,
            )
            previous_event_id = event.event_id
            previous_event_type = event.event_type
            frame = cv2.cvtColor(np.array(state.image.convert("RGB")), cv2.COLOR_RGB2BGR)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    if out is None:
        out = output_dir() / "aimtv-review.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Mux audio + video into a single reviewable MP4.
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(raw_video),
        "-i",
        str(plan.audio_wav),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{total_s:.6f}",
        "-shortest",
        str(out),
    ]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    manifest_out = out.with_suffix(".render.json")
    write_manifest(
        plan,
        manifest_out,
        output_video=str(out.resolve()),
        rendered_duration_s=round(total_s, 6),
        render={"fps": fps, "width": width, "height": height},
    )
    print(f"wrote review MP4: {out}", flush=True)
    print(f"wrote render manifest: {manifest_out}", flush=True)
    return out
