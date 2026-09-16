"""Deterministic contextual prompt scheduling over an Airadio audio timeline."""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from aimtv.context import ClipContext
from aimtv.keyphrases import corpus_idf, rank_keyphrases, words
from aimtv.playlist import Clip
from aimtv.prompts import PromptBank, pick_random_prompt

if TYPE_CHECKING:
    from aimtv.alignment import AlignmentResult, UnitAlignment

SECTION_RE = re.compile(r"^\[([^]]+)\]\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
MIN_CONTEXT_S = 6.0
MAX_PROMPT_WORDS = 60


@dataclass(frozen=True)
class LyricUnit:
    section: str
    text: str
    weight: int


@dataclass(frozen=True)
class PromptEvent:
    event_id: str
    start_s: float
    end_s: float
    event_type: str  # contextual | random-cutaway
    prompt: str
    hard_reset: bool
    clip_index: int | None
    clip_kind: str | None
    title: str | None
    section: str | None
    source_text: str | None
    keyphrases: tuple[str, ...]
    provenance_id: str | None
    lyrics_sha256: str | None
    verified: bool
    alignment_method: str | None
    alignment_confidence: float | None
    matched_words: tuple[str, ...]
    recognized_start_s: float | None
    recognized_end_s: float | None


@dataclass(frozen=True)
class PromptSchedule:
    seed: int
    total_s: float
    contextual: tuple[PromptEvent, ...]
    random_cutaways: tuple[PromptEvent, ...]

    def event_at(self, t: float) -> PromptEvent:
        for event in self.random_cutaways:
            if event.start_s <= t < event.end_s:
                return event
        for event in self.contextual:
            if event.start_s <= t < event.end_s:
                return event
        return self.contextual[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "seed": self.seed,
            "total_s": self.total_s,
            "contextual": [asdict(event) for event in self.contextual],
            "random_cutaways": [asdict(event) for event in self.random_cutaways],
        }


def parse_lyric_units(lyrics: str, *, kind: str) -> list[LyricUnit]:
    section = "interstitial" if kind != "song" else "song"
    units: list[LyricUnit] = []
    for raw in lyrics.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = SECTION_RE.match(line)
        if match:
            section = match.group(1).strip().lower()
            continue
        spans = SENTENCE_RE.split(line) if kind != "song" else [line]
        for span in spans:
            text = span.strip()
            if text:
                units.append(LyricUnit(section, text, max(1, len(words(text)))))
    return units


def _group_units(units: list[LyricUnit], duration_s: float) -> list[LyricUnit]:
    if not units:
        return []
    seconds_per_weight = duration_s / max(1, sum(unit.weight for unit in units))
    grouped: list[LyricUnit] = []
    buffer: list[LyricUnit] = []

    def flush() -> None:
        if not buffer:
            return
        grouped.append(
            LyricUnit(
                section=buffer[0].section,
                # Preserve source-line boundaries so RAKE never invents a phrase
                # by gluing the end of one lyric line to the next line.
                text="\n".join(unit.text for unit in buffer),
                weight=sum(unit.weight for unit in buffer),
            )
        )
        buffer.clear()

    for unit in units:
        if buffer and unit.section != buffer[0].section:
            flush()
        buffer.append(unit)
        if sum(item.weight for item in buffer) * seconds_per_weight >= MIN_CONTEXT_S:
            flush()
    flush()
    return grouped


def _visual_windows(starts: list[float], durs: list[float]) -> tuple[list[float], list[float]]:
    visual_starts = list(starts)
    visual_ends = [start + duration for start, duration in zip(starts, durs)]
    for index in range(1, len(visual_starts)):
        overlap = visual_ends[index - 1] - visual_starts[index]
        if overlap > 0:
            boundary = visual_starts[index] + overlap / 2.0
            visual_ends[index - 1] = boundary
            visual_starts[index] = boundary
    return visual_starts, visual_ends


def _trim_prompt(parts: list[str], max_words: int = MAX_PROMPT_WORDS) -> str:
    kept: list[str] = []
    count = 0
    for part in parts:
        part_words = words(part)
        if kept and count + len(part_words) > max_words:
            continue
        kept.append(part)
        count += len(part_words)
    return ", ".join(part for part in kept if part)


def _aligned_boundaries(
    units: list[LyricUnit],
    duration_s: float,
    alignment: AlignmentResult,
    *,
    kind: str,
) -> list[float]:
    """Warp proportional unit starts through reliable ASR timing anchors."""
    expected = [0.0]
    total_weight = max(1, sum(unit.weight for unit in units))
    cursor = 0
    for unit in units:
        cursor += unit.weight
        expected.append(duration_s * cursor / total_weight)

    anchors: list[tuple[int, float]] = [(0, 0.0), (len(units), duration_s)]
    for unit in alignment.units:
        if unit.unit_index >= len(units) or unit.start_s is None:
            continue
        if unit.unit_index in {0, len(units)}:
            continue
        enough_words = len(unit.matched_words) >= (1 if kind != "song" else 2)
        threshold = 0.20 if kind != "song" else 0.30
        if enough_words and unit.confidence >= threshold:
            # Anticipatory cuts feel synchronized; a late visual feels wrong.
            anchors.append((unit.unit_index, max(0.0, unit.start_s - 0.75)))

    by_index: dict[int, float] = {}
    for index, timing in sorted(anchors):
        by_index[index] = timing
    ordered: list[tuple[int, float]] = []
    for index, timing in sorted(by_index.items()):
        if ordered:
            timing = max(timing, ordered[-1][1] + 0.25)
        ordered.append((index, min(duration_s, timing)))
    ordered[-1] = (len(units), duration_s)

    boundaries = list(expected)
    for (left_i, left_t), (right_i, right_t) in zip(ordered, ordered[1:]):
        span = max(1e-6, expected[right_i] - expected[left_i])
        for index in range(left_i, right_i + 1):
            fraction = (expected[index] - expected[left_i]) / span
            boundaries[index] = left_t + (right_t - left_t) * fraction
    boundaries[0] = 0.0
    boundaries[-1] = duration_s
    for index in range(1, len(boundaries) - 1):
        boundaries[index] = min(
            duration_s,
            max(boundaries[index], boundaries[index - 1] + 0.25),
        )
    return boundaries


def _contextual_events(
    clips: list[Clip],
    contexts: list[ClipContext],
    starts: list[float],
    durs: list[float],
    bank: PromptBank,
    alignments: dict[str, AlignmentResult] | None,
) -> list[PromptEvent]:
    documents = [context.lyrics or context.title for context in contexts]
    idf = corpus_idf(documents)
    visual_starts, visual_ends = _visual_windows(starts, durs)
    events: list[PromptEvent] = []
    for clip_index, (clip, context, clip_start, clip_end) in enumerate(
        zip(clips, contexts, visual_starts, visual_ends)
    ):
        available = max(0.05, clip_end - clip_start)
        lyrics = context.lyrics if context.verified and context.lyrics else context.title
        raw_units = parse_lyric_units(lyrics, kind=clip.kind)
        alignment = (alignments or {}).get(str(clip.path.resolve()))
        aligned = alignment is not None and len(alignment.units) == len(raw_units)
        units = raw_units if aligned else _group_units(raw_units, available)
        if not units:
            units = [LyricUnit("title", context.title, 1)]
        global_phrases = [
            phrase
            for phrase, _ in rank_keyphrases(
                lyrics,
                idf=idf,
                title=context.title,
                recurrence_text=lyrics,
                limit=2,
            )
        ]
        if aligned and alignment is not None:
            local_boundaries = _aligned_boundaries(
                units,
                available,
                alignment,
                kind=clip.kind,
            )
        else:
            total_weight = max(1, sum(unit.weight for unit in units))
            cumulative = 0
            local_boundaries = [0.0]
            for unit in units:
                cumulative += unit.weight
                local_boundaries.append(available * cumulative / total_weight)
        for unit_index, unit in enumerate(units):
            cursor = clip_start + local_boundaries[unit_index]
            end = clip_start + local_boundaries[unit_index + 1]
            unit_alignment: UnitAlignment | None = None
            if aligned and alignment is not None:
                unit_alignment = alignment.units[unit_index]
            local = [
                phrase
                for phrase, _ in rank_keyphrases(
                    unit.text,
                    idf=idf,
                    title=context.title,
                    recurrence_text=lyrics,
                    limit=3,
                )
            ]
            title_phrase = [context.title.lower()] if clip.kind == "song" else []
            phrases = list(dict.fromkeys(local + global_phrases + title_phrase))
            prompt = _trim_prompt(phrases + [bank.style_suffix])
            section_changed = unit_index == 0 or unit.section != units[unit_index - 1].section
            events.append(
                PromptEvent(
                    event_id=f"context-{clip_index:04d}-{unit_index:04d}",
                    start_s=round(cursor, 6),
                    end_s=round(end, 6),
                    event_type="contextual",
                    prompt=prompt,
                    hard_reset=section_changed,
                    clip_index=clip_index,
                    clip_kind=clip.kind,
                    title=context.title,
                    section=unit.section,
                    source_text=unit.text if context.verified else None,
                    keyphrases=tuple(phrases),
                    provenance_id=context.provenance_id,
                    lyrics_sha256=context.lyrics_sha256,
                    verified=context.verified,
                    alignment_method=(
                        unit_alignment.method if unit_alignment is not None else "proportional"
                    ),
                    alignment_confidence=(
                        unit_alignment.confidence if unit_alignment is not None else None
                    ),
                    matched_words=(
                        unit_alignment.matched_words if unit_alignment is not None else ()
                    ),
                    recognized_start_s=(
                        round(starts[clip_index] + unit_alignment.start_s, 6)
                        if unit_alignment is not None and unit_alignment.start_s is not None
                        else None
                    ),
                    recognized_end_s=(
                        round(starts[clip_index] + unit_alignment.end_s, 6)
                        if unit_alignment is not None and unit_alignment.end_s is not None
                        else None
                    ),
                )
            )
    return events


def _subtract_intervals(
    window: tuple[float, float], protected: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    pieces = [window]
    for block_start, block_end in sorted(protected):
        next_pieces: list[tuple[float, float]] = []
        for start, end in pieces:
            if block_end <= start or block_start >= end:
                next_pieces.append((start, end))
                continue
            if block_start > start:
                next_pieces.append((start, block_start))
            if block_end < end:
                next_pieces.append((block_end, end))
        pieces = next_pieces
    return [(start, end) for start, end in pieces if end - start >= 2.0]


def _protected_song_anchors(
    events: list[PromptEvent], clip_index: int
) -> list[tuple[float, float]]:
    candidates = [
        event
        for event in events
        if event.clip_index == clip_index
        and event.alignment_confidence is not None
        and event.alignment_confidence >= 0.30
        and len(event.matched_words) >= 2
    ]
    chosen: list[PromptEvent] = []
    for event in sorted(
        candidates,
        key=lambda item: (-(item.alignment_confidence or 0.0), -len(item.matched_words)),
    ):
        center = (event.start_s + event.end_s) / 2.0
        if any(abs(center - (old.start_s + old.end_s) / 2.0) < 7.0 for old in chosen):
            continue
        chosen.append(event)
        if len(chosen) >= 6:
            break
    protected: list[tuple[float, float]] = []
    for event in chosen:
        if event.recognized_start_s is None:
            continue
        start = max(event.start_s, event.recognized_start_s - 0.75)
        spoken_end = event.recognized_end_s or event.recognized_start_s
        end = min(event.end_s, max(spoken_end + 2.5, start + 4.0))
        if end > start:
            protected.append((start, end))
    return protected


def _song_dream_cutaways(
    clips: list[Clip],
    starts: list[float],
    durs: list[float],
    contextual: list[PromptEvent],
    rng: random.Random,
    bank: PromptBank,
    *,
    min_gap_s: float,
    max_gap_s: float,
    min_duration_s: float,
    max_duration_s: float,
) -> list[PromptEvent]:
    events: list[PromptEvent] = []
    visual_starts, visual_ends = _visual_windows(starts, durs)
    for clip_index, clip in enumerate(clips):
        if clip.kind != "song":
            continue
        protected = _protected_song_anchors(contextual, clip_index)
        eligible = _subtract_intervals(
            (visual_starts[clip_index], visual_ends[clip_index]), protected
        )
        for interval_start, interval_end in eligible:
            cursor = interval_start + rng.uniform(min_gap_s, max_gap_s)
            while cursor < interval_end - 1.0:
                duration = min(
                    rng.uniform(min_duration_s, max_duration_s),
                    interval_end - cursor,
                )
                if duration < 2.0:
                    break
                prompt, subject = pick_random_prompt(rng, bank)
                events.append(
                    PromptEvent(
                        event_id=f"random-{len(events):04d}",
                        start_s=round(cursor, 6),
                        end_s=round(cursor + duration, 6),
                        event_type="random-cutaway",
                        prompt=prompt,
                        hard_reset=True,
                        clip_index=clip_index,
                        clip_kind="song",
                        title=clip.title,
                        section=None,
                        source_text=None,
                        keyphrases=(subject,),
                        provenance_id=None,
                        lyrics_sha256=None,
                        verified=True,
                        alignment_method="deliberate-dream-cutaway",
                        alignment_confidence=None,
                        matched_words=(),
                        recognized_start_s=None,
                        recognized_end_s=None,
                    )
                )
                cursor += duration + rng.uniform(min_gap_s, max_gap_s)
    return events


def build_prompt_schedule(
    clips: list[Clip],
    contexts: list[ClipContext],
    starts: list[float],
    durs: list[float],
    bank: PromptBank,
    *,
    seed: int,
    min_random_gap_s: float = 3.0,
    max_random_gap_s: float = 8.0,
    min_random_duration_s: float = 7.0,
    max_random_duration_s: float = 16.0,
    alignments: dict[str, AlignmentResult] | None = None,
) -> PromptSchedule:
    if (
        not clips
        or len(clips) != len(contexts)
        or len(clips) != len(starts)
        or len(clips) != len(durs)
    ):
        raise ValueError("clips, contexts, starts, and durations must be non-empty and aligned")
    if min_random_gap_s <= 0 or max_random_gap_s < min_random_gap_s:
        raise ValueError("invalid random cutaway gap")
    if min_random_duration_s <= 0 or max_random_duration_s < min_random_duration_s:
        raise ValueError("invalid random cutaway duration")
    contextual = _contextual_events(clips, contexts, starts, durs, bank, alignments)
    total_s = max(event.end_s for event in contextual)
    rng = random.Random(seed ^ 0xA1F00D)
    random_events = _song_dream_cutaways(
        clips,
        starts,
        durs,
        contextual,
        rng,
        bank,
        min_gap_s=min_random_gap_s,
        max_gap_s=max_random_gap_s,
        min_duration_s=min_random_duration_s,
        max_duration_s=max_random_duration_s,
    )
    return PromptSchedule(
        seed=seed,
        total_s=round(total_s, 6),
        contextual=tuple(contextual),
        random_cutaways=tuple(random_events),
    )
