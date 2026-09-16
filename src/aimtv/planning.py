"""Build a provenance-aware AI MTV timeline without loading the vision model."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aimtv import __version__
from aimtv.alignment import AlignmentResult, load_cached_alignments
from aimtv.audio_ops import stitch_like_airadio
from aimtv.context import ClipContext, resolve_contexts
from aimtv.paths import airadio_home, ensure_aimtv_layout, output_dir
from aimtv.playlist import Clip, PlexConfig, build_review_playlist
from aimtv.prompts import load_prompt_bank
from aimtv.schedule import PromptSchedule, build_prompt_schedule


@dataclass(frozen=True)
class ReviewPlan:
    seed: int
    clips: tuple[Clip, ...]
    contexts: tuple[ClipContext, ...]
    starts: tuple[float, ...]
    durations: tuple[float, ...]
    audio_wav: Path
    schedule: PromptSchedule
    manifest_path: Path
    alignments: dict[str, AlignmentResult]


def _clip_manifest(
    index: int,
    clip: Clip,
    context: ClipContext,
    start: float,
    duration: float,
    alignment: AlignmentResult | None,
) -> dict[str, Any]:
    return {
        "index": index,
        "audio": str(clip.path.resolve()),
        "kind": clip.kind,
        "title": context.title,
        "start_s": round(start, 6),
        "duration_s": round(duration, 6),
        "context": {
            "verified": context.verified,
            "source": context.source,
            "provenance_id": context.provenance_id,
            "lyrics_sha256": context.lyrics_sha256,
            "warning": context.warning,
        },
        "voice_alignment": alignment.to_manifest() if alignment is not None else None,
    }


def manifest_data(plan: ReviewPlan, **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "manifest_version": 2,
        "aimtv_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": plan.seed,
        "airadio_home": str(airadio_home()),
        "audio_timeline": str(plan.audio_wav.resolve()),
        "prompt_bank": str(load_prompt_bank().source.resolve()),
        "clips": [
            _clip_manifest(
                index,
                clip,
                context,
                start,
                duration,
                plan.alignments.get(str(clip.path.resolve())),
            )
            for index, (clip, context, start, duration) in enumerate(
                zip(plan.clips, plan.contexts, plan.starts, plan.durations)
            )
        ],
        "schedule": plan.schedule.to_dict(),
    }
    data.update(extra)
    return data


def write_manifest(plan: ReviewPlan, path: Path | None = None, **extra: Any) -> Path:
    dest = path or plan.manifest_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp")
    tmp.write_text(
        json.dumps(manifest_data(plan, **extra), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(dest)
    return dest


def build_review_plan(
    *,
    song_count: int = 2,
    interstitial_min: int = 1,
    interstitial_max: int = 3,
    seed: int | None = None,
    manifest_path: Path | None = None,
    min_random_gap_s: float = 3.0,
    max_random_gap_s: float = 8.0,
    min_random_duration_s: float = 7.0,
    max_random_duration_s: float = 16.0,
    plex: PlexConfig | None = None,
) -> ReviewPlan:
    """Select, stitch, verify, schedule, and record a finite review timeline."""
    ensure_aimtv_layout()
    render_seed = int(seed if seed is not None else secrets.randbits(32))
    clips = build_review_playlist(
        song_count=song_count,
        interstitial_min=interstitial_min,
        interstitial_max=interstitial_max,
        seed=render_seed,
        plex=plex,
    )

    staging = output_dir() / "review-staging"
    staging.mkdir(parents=True, exist_ok=True)
    audio_wav = staging / "timeline.wav"
    _, starts, durations = stitch_like_airadio(
        clips, staging / "airadio-stitch", audio_wav
    )
    contexts = resolve_contexts(clips, airadio_home())
    alignments = load_cached_alignments(clips, contexts)
    schedule = build_prompt_schedule(
        clips,
        contexts,
        starts,
        durations,
        load_prompt_bank(),
        seed=render_seed,
        min_random_gap_s=min_random_gap_s,
        max_random_gap_s=max_random_gap_s,
        min_random_duration_s=min_random_duration_s,
        max_random_duration_s=max_random_duration_s,
        alignments=alignments,
    )
    plan = ReviewPlan(
        seed=render_seed,
        clips=tuple(clips),
        contexts=tuple(contexts),
        starts=tuple(starts),
        durations=tuple(durations),
        audio_wav=audio_wav,
        schedule=schedule,
        manifest_path=manifest_path or staging / "render-manifest.json",
        alignments=alignments,
    )
    write_manifest(plan)
    return plan


def apply_alignments(
    plan: ReviewPlan,
    alignments: dict[str, AlignmentResult],
    *,
    min_random_gap_s: float = 3.0,
    max_random_gap_s: float = 8.0,
    min_random_duration_s: float = 7.0,
    max_random_duration_s: float = 16.0,
) -> ReviewPlan:
    """Rebuild timing and dream cutaways from newly generated alignment evidence."""
    schedule = build_prompt_schedule(
        list(plan.clips),
        list(plan.contexts),
        list(plan.starts),
        list(plan.durations),
        load_prompt_bank(),
        seed=plan.seed,
        min_random_gap_s=min_random_gap_s,
        max_random_gap_s=max_random_gap_s,
        min_random_duration_s=min_random_duration_s,
        max_random_duration_s=max_random_duration_s,
        alignments=alignments,
    )
    updated = replace(plan, schedule=schedule, alignments=dict(alignments))
    write_manifest(updated)
    return updated
