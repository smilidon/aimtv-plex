"""Small deterministic RAKE/IDF keyphrase extractor for lyric imagery."""

from __future__ import annotations

import math
import re
from collections import Counter

WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?", re.IGNORECASE)
BOUNDARY_RE = re.compile(r"[,.;:!?(){}\[\]/\\|\n]+")

# Function words split phrase candidates. This intentionally stays local and
# transparent instead of introducing a statistical model or remote service.
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be because
    been before being below between both but by can can't cannot could couldn't did
    didn't do does doesn't doing don't down during each few for from further had
    hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
    herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's
    its itself just let's me more most mustn't my myself no nor not of off on once
    only or other ought our ours ourselves out over own same shan't she she'd she'll
    she's should shouldn't so some such than that that's the their theirs them
    themselves then there there's these they they'd they'll they're they've this
    those through to too under until up very was wasn't we we'd we'll we're we've were
    weren't what what's when when's where where's which while who who's whom why why's
    with won't would wouldn't you you'd you'll you're you've your yours yourself
    yourselves yeah oh ooh whoa la na try verse chorus bridge intro outro pre post
    """.split()
)


def words(text: str) -> list[str]:
    return [token.lower().replace("’", "'") for token in WORD_RE.findall(text)]


def _candidate_chunks(text: str, *, max_words: int = 5) -> list[tuple[str, ...]]:
    chunks: list[tuple[str, ...]] = []
    for span in BOUNDARY_RE.split(text.lower().replace("’", "'")):
        current: list[str] = []
        for token in words(span):
            if token in STOPWORDS or len(token) < 2 or token.isdigit():
                if current:
                    chunks.extend(_split_long_chunk(current, max_words))
                    current = []
                continue
            current.append(token)
        if current:
            chunks.extend(_split_long_chunk(current, max_words))
    return chunks


def _split_long_chunk(chunk: list[str], max_words: int) -> list[tuple[str, ...]]:
    if len(chunk) <= max_words:
        return [tuple(chunk)]
    # Preserve neighboring words while keeping prompts within CLIP's small token budget.
    return [tuple(chunk[i : i + max_words]) for i in range(0, len(chunk), max_words)]


def corpus_idf(documents: list[str]) -> dict[str, float]:
    count = max(1, len(documents))
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(words(document)) - STOPWORDS)
    return {
        token: math.log((count + 1.0) / (frequency + 1.0)) + 1.0
        for token, frequency in document_frequency.items()
    }


def rank_keyphrases(
    text: str,
    *,
    idf: dict[str, float],
    title: str = "",
    recurrence_text: str | None = None,
    limit: int = 3,
) -> list[tuple[str, float]]:
    candidates = _candidate_chunks(text)
    if not candidates:
        fallback = [token for token in words(text) if token not in STOPWORDS]
        return [(token, idf.get(token, 1.0)) for token in fallback[:limit]]

    frequency: Counter[str] = Counter()
    degree: Counter[str] = Counter()
    for phrase in candidates:
        phrase_degree = max(0, len(phrase) - 1)
        for token in phrase:
            frequency[token] += 1
            degree[token] += phrase_degree
    word_score = {
        token: (degree[token] + frequency[token]) / frequency[token]
        for token in frequency
    }
    title_tokens = set(words(title)) - STOPWORDS
    recurrence_source = " ".join(words(recurrence_text or text))
    scored: dict[str, float] = {}
    for phrase in candidates:
        label = " ".join(phrase)
        rake = sum(word_score[token] for token in phrase)
        rarity = sum(idf.get(token, 1.0) for token in phrase) / len(phrase)
        title_bonus = 1.0 + 0.25 * len(set(phrase) & title_tokens)
        occurrences = max(1, recurrence_source.count(label))
        recurrence_bonus = 1.0 + min(occurrences - 1, 3) * 0.18
        score = rake * rarity * title_bonus * recurrence_bonus
        scored[label] = max(score, scored.get(label, 0.0))

    selected: list[tuple[str, float]] = []
    for label, score in sorted(scored.items(), key=lambda item: (-item[1], item[0])):
        tokens = set(label.split())
        if any(tokens <= set(existing.split()) or set(existing.split()) <= tokens for existing, _ in selected):
            continue
        selected.append((label, score))
        if len(selected) >= limit:
            break
    return selected
