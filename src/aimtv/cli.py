"""aimtv CLI."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from aimtv import __version__
from aimtv.playlist import PlexConfig
from aimtv.preflight import check_airadio_ready, preflight


def _add_playlist_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--songs",
        type=int,
        default=2,
        help="library songs in the review playlist (default: 2)",
    )
    parser.add_argument(
        "--interstitial-min",
        type=int,
        default=1,
        help="min interstitials between songs (default: 1)",
    )
    parser.add_argument(
        "--interstitial-max",
        type=int,
        default=3,
        help="max interstitials between songs (default: 3)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--random-min-gap", type=float, default=3.0)
    parser.add_argument("--random-max-gap", type=float, default=8.0)
    parser.add_argument("--random-min-seconds", type=float, default=7.0)
    parser.add_argument("--random-max-seconds", type=float, default=16.0)
    plex = parser.add_argument_group(
        "plex source",
        "Take songs from a Plex music section instead of the Airadio library. "
        "Interstitials still come from Airadio. Each flag falls back to the "
        "environment variable in parentheses; prefer the variables for tokens.",
    )
    plex.add_argument("--plex-url", default=None, help="Plex server URL, e.g. http://localhost:32400 (PLEX_URL)")
    plex.add_argument("--plex-token", default=None, help="Plex auth token (PLEX_TOKEN)")
    plex.add_argument("--plex-section", default=None, help="Music library section name, e.g. Music (PLEX_SECTION)")
    plex.add_argument("--genius-token", default=None, help="Genius API token for lyric lookup when tags have none (GENIUS_TOKEN)")


def _plex_config(args: argparse.Namespace) -> PlexConfig | None:
    """Build a PlexConfig from flags/env, or None when no Plex option is set."""
    url = args.plex_url or os.environ.get("PLEX_URL")
    token = args.plex_token or os.environ.get("PLEX_TOKEN")
    section = args.plex_section or os.environ.get("PLEX_SECTION")
    genius = args.genius_token or os.environ.get("GENIUS_TOKEN")
    if not any((url, token, section)):
        return None
    missing = [name for name, value in (("--plex-url/PLEX_URL", url), ("--plex-token/PLEX_TOKEN", token), ("--plex-section/PLEX_SECTION", section)) if not value]
    if missing:
        raise SystemExit(f"Plex source needs all of url, token and section; missing: {', '.join(missing)}")
    return PlexConfig(url=str(url), token=str(token), section=str(section), genius_token=genius or None)


def _playlist_kwargs(args: argparse.Namespace) -> dict:
    imin = max(0, int(args.interstitial_min))
    imax = max(imin, int(args.interstitial_max))
    return {
        "song_count": max(1, int(args.songs)),
        "interstitial_min": imin,
        "interstitial_max": imax,
        "seed": args.seed,
        "min_random_gap_s": float(args.random_min_gap),
        "max_random_gap_s": float(args.random_max_gap),
        "min_random_duration_s": float(args.random_min_seconds),
        "max_random_duration_s": float(args.random_max_seconds),
        "plex": _plex_config(args),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aimtv",
        description="AI MTV — AI Music Television from a local Airadio library or Plex server",
    )
    parser.add_argument("--version", action="version", version=f"aimtv {__version__}")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="run aimtv (default: review MP4 of 2 songs)")
    run_p.add_argument(
        "--review",
        action="store_true",
        default=True,
        help="finite review build (2 library songs + interstitials → one MP4)",
    )
    _add_playlist_args(run_p)
    run_p.add_argument("--fps", type=float, default=8.0)
    run_p.add_argument("--width", type=int, default=512)
    run_p.add_argument("--height", type=int, default=320)
    run_p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="render only the opening N seconds (useful for a quick preview)",
    )
    run_p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output MP4 path (default: ~/.local/share/aimtv/output/aimtv-review.mp4)",
    )
    run_p.add_argument(
        "--yes",
        action="store_true",
        help="allow model download without an interactive prompt",
    )
    run_p.add_argument(
        "--no-voice-alignment",
        action="store_true",
        help="skip local ASR timing and use proportional lyric timing",
    )

    plan_p = sub.add_parser(
        "plan",
        help="build audio, verify provenance, and write prompts without loading the GPU",
    )
    _add_playlist_args(plan_p)
    plan_p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest path (default: AI MTV review staging directory)",
    )

    sub.add_parser("doctor", help="run preflight checks only")

    fetch_p = sub.add_parser("fetch-models", help="download models with consent + progress")
    fetch_p.add_argument("--yes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        raw_args = ["run"]
    args = parser.parse_args(raw_args)
    command = args.command or "run"

    if command == "doctor":
        return preflight()

    if command == "fetch-models":
        if check_airadio_ready() != 0:
            return 1
        from aimtv.models import fetch_required_models

        try:
            fetch_required_models(assume_yes=bool(args.yes))
        except SystemExit as exc:
            return int(exc.code) if isinstance(exc.code, int) else 1
        return 0

    if command == "run":
        # Keep this lightweight check before importing the Airadio-dependent
        # rendering modules, so a missing dependency produces useful guidance.
        # Plex mode still needs Airadio: interstitials and provenance come from it.
        if check_airadio_ready() != 0:
            return 1
        from aimtv.review import render_review_mp4

        render_review_mp4(
            out=args.out,
            fps=args.fps,
            width=args.width,
            height=args.height,
            assume_yes_models=bool(args.yes),
            max_seconds=args.max_seconds,
            voice_alignment=not bool(args.no_voice_alignment),
            **_playlist_kwargs(args),
        )
        return 0

    if command == "plan":
        if check_airadio_ready() != 0:
            return 1
        from aimtv.planning import build_review_plan

        plan = build_review_plan(manifest_path=args.manifest, **_playlist_kwargs(args))
        print(f"review seed: {plan.seed}")
        for clip, context in zip(plan.clips, plan.contexts):
            status = "verified" if context.verified else f"UNVERIFIED: {context.warning}"
            print(f"  [{clip.kind}] {context.title} ({status})")
        print(f"wrote plan manifest: {plan.manifest_path}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
