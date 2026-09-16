"""Local voice recognition used only as timing evidence for verified lyrics."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from aimtv.context import ClipContext
from aimtv.keyphrases import STOPWORDS, words
from aimtv.paths import alignments_dir, ensure_aimtv_layout, models_dir
from aimtv.playlist import Clip
from aimtv.schedule import parse_lyric_units

ALIGNMENT_SCHEMA = 1
ASR_MODEL_ID = "Systran/faster-whisper-small"
ASR_MODEL_DIR = "faster-whisper-small"
ASR_APPROX_GB = 0.5


@dataclass(frozen=True)
class RecognizedWord:
    text: str
    start_s: float
    end_s: float
    probability: float


@dataclass(frozen=True)
class UnitAlignment:
    unit_index: int
    start_s: float | None
    end_s: float | None
    confidence: float
    matched_words: tuple[str, ...]
    method: str


@dataclass(frozen=True)
class AlignmentResult:
    schema_version: int
    audio_sha256: str
    lyrics_sha256: str
    model_id: str
    recognized_text: str
    matched_token_ratio: float
    units: tuple[UnitAlignment, ...]
    cache_path: Path

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "audio_sha256": self.audio_sha256,
            "lyrics_sha256": self.lyrics_sha256,
            "model_id": self.model_id,
            "matched_token_ratio": self.matched_token_ratio,
            "recognized_text": self.recognized_text,
            "cache_path": str(self.cache_path.resolve()),
            "units": [asdict(unit) for unit in self.units],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alignment_model_root() -> Path:
    ensure_aimtv_layout()
    return models_dir() / ASR_MODEL_DIR


def alignment_model_present(root: Path | None = None) -> bool:
    path = root or alignment_model_root()
    required = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")
    return all((path / name).is_file() for name in required)


def _ask_model_permission() -> bool:
    print(
        "\nAI MTV voice timing needs one local speech-recognition model.\n"
        f"  Model:   {ASR_MODEL_ID}\n"
        f"  Dest:    {alignment_model_root()}\n"
        f"  Size:    ~{ASR_APPROX_GB:.1f} GB\n"
        "  Use:     local timing evidence only; no audio or lyrics are uploaded\n",
        flush=True,
    )
    try:
        answer = input("Allow AI MTV to download this model? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    return answer in {"y", "yes"}


def fetch_alignment_model(*, assume_yes: bool = False) -> Path:
    dest = alignment_model_root()
    if alignment_model_present(dest):
        return dest
    if not assume_yes and not _ask_model_permission():
        raise SystemExit(1)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {ASR_MODEL_ID} for local voice timing → {dest}", flush=True)
    snapshot_download(
        repo_id=ASR_MODEL_ID,
        local_dir=str(dest),
        allow_patterns=("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"),
    )
    if not alignment_model_present(dest):
        raise RuntimeError(f"incomplete voice-timing model under {dest}")
    return dest


def _cache_path(audio_hash: str, lyrics_hash: str) -> Path:
    ensure_aimtv_layout()
    return alignments_dir() / f"{audio_hash[:20]}-{lyrics_hash[:20]}.json"


def _from_json(path: Path, raw: dict[str, Any]) -> AlignmentResult:
    return AlignmentResult(
        schema_version=int(raw["schema_version"]),
        audio_sha256=str(raw["audio_sha256"]),
        lyrics_sha256=str(raw["lyrics_sha256"]),
        model_id=str(raw["model_id"]),
        recognized_text=str(raw.get("recognized_text") or ""),
        matched_token_ratio=float(raw.get("matched_token_ratio") or 0.0),
        units=tuple(
            UnitAlignment(
                unit_index=int(unit["unit_index"]),
                start_s=float(unit["start_s"]) if unit.get("start_s") is not None else None,
                end_s=float(unit["end_s"]) if unit.get("end_s") is not None else None,
                confidence=float(unit.get("confidence") or 0.0),
                matched_words=tuple(str(word) for word in unit.get("matched_words", [])),
                method=str(unit.get("method") or "none"),
            )
            for unit in raw.get("units", [])
        ),
        cache_path=path,
    )


def load_cached_alignment(clip: Clip, context: ClipContext) -> AlignmentResult | None:
    if not context.verified or not context.lyrics or not context.lyrics_sha256:
        return None
    audio_hash = _sha256_file(clip.path)
    path = _cache_path(audio_hash, context.lyrics_sha256)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        result = _from_json(path, raw)
    except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if (
        result.schema_version != ALIGNMENT_SCHEMA
        or result.audio_sha256 != audio_hash
        or result.lyrics_sha256 != context.lyrics_sha256
        or result.model_id != ASR_MODEL_ID
    ):
        return None
    return result


def load_cached_alignments(
    clips: list[Clip] | tuple[Clip, ...],
    contexts: list[ClipContext] | tuple[ClipContext, ...],
) -> dict[str, AlignmentResult]:
    found: dict[str, AlignmentResult] = {}
    for clip, context in zip(clips, contexts):
        result = load_cached_alignment(clip, context)
        if result is not None:
            found[str(clip.path.resolve())] = result
    return found


def _token_similarity(left: str, right: str) -> float:
    if left == right:
        return 3.0
    if len(left) >= 4 and len(right) >= 4 and SequenceMatcher(None, left, right).ratio() >= 0.82:
        return 1.25
    return -1.5


def _monotonic_matches(lyrics: list[str], recognized: list[str]) -> list[tuple[int, int]]:
    """Needleman-Wunsch alignment; return only exact or strong fuzzy matches."""
    n = len(lyrics)
    m = len(recognized)
    gap = -0.55
    scores = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]  # 1 diagonal, 2 up, 3 left
    for i in range(1, n + 1):
        scores[i][0] = i * gap
        trace[i][0] = 2
    for j in range(1, m + 1):
        scores[0][j] = j * gap
        trace[0][j] = 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            similarity = _token_similarity(lyrics[i - 1], recognized[j - 1])
            candidates = (
                (scores[i - 1][j - 1] + similarity, 1),
                (scores[i - 1][j] + gap, 2),
                (scores[i][j - 1] + gap, 3),
            )
            scores[i][j], trace[i][j] = max(candidates, key=lambda item: item[0])
    matches: list[tuple[int, int]] = []
    i, j = n, m
    while i or j:
        direction = trace[i][j]
        if direction == 1:
            if _token_similarity(lyrics[i - 1], recognized[j - 1]) > 0:
                matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif direction == 2:
            i -= 1
        else:
            j -= 1
    matches.reverse()
    return matches


def align_words_to_lyrics(
    lyrics: str,
    *,
    kind: str,
    recognized_words: list[RecognizedWord],
    audio_sha256: str,
    lyrics_sha256: str,
    cache_path: Path,
) -> AlignmentResult:
    units = parse_lyric_units(lyrics, kind=kind)
    lyric_tokens: list[str] = []
    token_units: list[int] = []
    unit_content_counts: list[int] = []
    for index, unit in enumerate(units):
        tokens = words(unit.text)
        lyric_tokens.extend(tokens)
        token_units.extend([index] * len(tokens))
        content = [token for token in tokens if token not in STOPWORDS]
        unit_content_counts.append(max(1, len(content)))

    recognized_tokens: list[str] = []
    recognized_refs: list[int] = []
    for index, word in enumerate(recognized_words):
        for token in words(word.text):
            recognized_tokens.append(token)
            recognized_refs.append(index)

    matches = _monotonic_matches(lyric_tokens, recognized_tokens)
    per_unit: list[list[tuple[int, int]]] = [[] for _ in units]
    for lyric_index, recognized_index in matches:
        per_unit[token_units[lyric_index]].append((lyric_index, recognized_index))

    aligned_units: list[UnitAlignment] = []
    matched_content_total = 0
    content_total = sum(unit_content_counts)
    for unit_index, unit_matches in enumerate(per_unit):
        refs = sorted({recognized_refs[recognized_index] for _, recognized_index in unit_matches})
        evidence = [recognized_words[index] for index in refs]
        matched_content = {
            lyric_tokens[lyric_index]
            for lyric_index, _ in unit_matches
            if lyric_tokens[lyric_index] not in STOPWORDS
        }
        matched_content_total += len(matched_content)
        coverage = min(1.0, len(matched_content) / unit_content_counts[unit_index])
        probability = (
            sum(word.probability for word in evidence) / len(evidence) if evidence else 0.0
        )
        confidence = coverage * (0.5 + 0.5 * max(0.0, min(1.0, probability)))
        aligned_units.append(
            UnitAlignment(
                unit_index=unit_index,
                start_s=min((word.start_s for word in evidence), default=None),
                end_s=max((word.end_s for word in evidence), default=None),
                confidence=round(confidence, 6),
                matched_words=tuple(word.text.strip() for word in evidence),
                method="voice-asr-monotonic" if evidence else "unmatched",
            )
        )
    return AlignmentResult(
        schema_version=ALIGNMENT_SCHEMA,
        audio_sha256=audio_sha256,
        lyrics_sha256=lyrics_sha256,
        model_id=ASR_MODEL_ID,
        recognized_text=" ".join(word.text.strip() for word in recognized_words).strip(),
        matched_token_ratio=round(matched_content_total / max(1, content_total), 6),
        units=tuple(aligned_units),
        cache_path=cache_path,
    )


def _write_alignment(result: AlignmentResult) -> None:
    path = result.cache_path
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = result.to_manifest()
    raw.pop("cache_path", None)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def ensure_alignments(
    clips: list[Clip] | tuple[Clip, ...],
    contexts: list[ClipContext] | tuple[ClipContext, ...],
    *,
    assume_yes_models: bool = False,
) -> dict[str, AlignmentResult]:
    results = load_cached_alignments(clips, contexts)
    missing = [
        (clip, context)
        for clip, context in zip(clips, contexts)
        if context.verified
        and context.lyrics
        and context.lyrics_sha256
        and str(clip.path.resolve()) not in results
    ]
    if not missing:
        return results

    from faster_whisper import WhisperModel

    model_path = fetch_alignment_model(assume_yes=assume_yes_models)
    threads = max(1, min(8, (os.cpu_count() or 4) // 2))
    print(f"loading local voice-timing model on CPU-int8 ({threads} threads)…", flush=True)
    model = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=threads)
    for clip, context in missing:
        print(f"voice timing: [{clip.kind}] {context.title}", flush=True)
        segments, _ = model.transcribe(
            str(clip.path),
            language="en",
            task="transcribe",
            beam_size=5,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        recognized: list[RecognizedWord] = []
        for segment in segments:
            for word in segment.words or ():
                recognized.append(
                    RecognizedWord(
                        text=str(word.word),
                        start_s=float(word.start),
                        end_s=float(word.end),
                        probability=float(word.probability),
                    )
                )
        audio_hash = _sha256_file(clip.path)
        cache_path = _cache_path(audio_hash, str(context.lyrics_sha256))
        result = align_words_to_lyrics(
            str(context.lyrics),
            kind=clip.kind,
            recognized_words=recognized,
            audio_sha256=audio_hash,
            lyrics_sha256=str(context.lyrics_sha256),
            cache_path=cache_path,
        )
        _write_alignment(result)
        results[str(clip.path.resolve())] = result
        anchors = sum(
            1
            for unit in result.units
            if unit.confidence >= 0.30 and len(unit.matched_words) >= 2
        )
        print(
            f"  matched content {result.matched_token_ratio:.0%}; convincing units {anchors}",
            flush=True,
        )
    return results
