from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aimtv.alignment import RecognizedWord, align_words_to_lyrics


class VoiceAlignmentTests(unittest.TestCase):
    def test_recognized_words_are_monotonically_matched_to_verified_lines(self) -> None:
        lyrics = (
            "[Verse]\n"
            "A neon hedgehog crosses Jakarta\n"
            "Cyberpunk rain returns tonight\n"
            "[Chorus]\n"
            "Expert systems call Pandu home"
        )
        recognized = [
            RecognizedWord("neon", 8.0, 8.4, 0.95),
            RecognizedWord("hedgehog", 8.4, 9.0, 0.93),
            RecognizedWord("crosses", 9.0, 9.5, 0.90),
            RecognizedWord("Jakarta", 9.5, 10.1, 0.96),
            RecognizedWord("cyberpunk", 31.0, 31.8, 0.91),
            RecognizedWord("rain", 31.8, 32.2, 0.94),
            RecognizedWord("returns", 32.2, 32.8, 0.88),
            RecognizedWord("expert", 54.0, 54.5, 0.90),
            RecognizedWord("systems", 54.5, 55.1, 0.92),
            RecognizedWord("Pandu", 56.0, 56.5, 0.96),
        ]
        with tempfile.TemporaryDirectory() as raw:
            result = align_words_to_lyrics(
                lyrics,
                kind="song",
                recognized_words=recognized,
                audio_sha256="a" * 64,
                lyrics_sha256="b" * 64,
                cache_path=Path(raw) / "alignment.json",
            )
        self.assertEqual(len(result.units), 3)
        self.assertEqual(result.units[0].start_s, 8.0)
        self.assertEqual(result.units[1].start_s, 31.0)
        self.assertEqual(result.units[2].start_s, 54.0)
        self.assertGreater(result.units[0].confidence, 0.7)
        self.assertGreater(result.matched_token_ratio, 0.5)

    def test_unheard_lines_are_not_given_fabricated_timestamps(self) -> None:
        lyrics = "[Verse]\nFirst silver river\nSecond copper cloud"
        recognized = [RecognizedWord("silver", 4.0, 4.5, 0.9)]
        with tempfile.TemporaryDirectory() as raw:
            result = align_words_to_lyrics(
                lyrics,
                kind="song",
                recognized_words=recognized,
                audio_sha256="a" * 64,
                lyrics_sha256="b" * 64,
                cache_path=Path(raw) / "alignment.json",
            )
        self.assertEqual(result.units[0].start_s, 4.0)
        self.assertIsNone(result.units[1].start_s)
        self.assertEqual(result.units[1].method, "unmatched")


if __name__ == "__main__":
    unittest.main()
