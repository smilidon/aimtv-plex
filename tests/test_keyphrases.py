from __future__ import annotations

import unittest

from aimtv.keyphrases import corpus_idf, rank_keyphrases


class KeyphraseTests(unittest.TestCase):
    def test_distinctive_phrases_rank_deterministically(self) -> None:
        documents = [
            "Try new crab milk. Cyberpunk Jakarta glows after midnight.",
            "A pale river crosses the ordinary city.",
        ]
        idf = corpus_idf(documents)
        first = rank_keyphrases(
            documents[0], idf=idf, recurrence_text=documents[0], limit=4
        )
        second = rank_keyphrases(
            documents[0], idf=idf, recurrence_text=documents[0], limit=4
        )
        self.assertEqual(first, second)
        labels = [label for label, _ in first]
        self.assertIn("new crab milk", labels)
        self.assertTrue(any("cyberpunk jakarta" in label for label in labels))

    def test_stopwords_split_candidates(self) -> None:
        ranked = rank_keyphrases(
            "the hedgehog and the expert systems under Jakarta",
            idf={},
            limit=5,
        )
        labels = [label for label, _ in ranked]
        self.assertIn("hedgehog", labels)
        self.assertIn("expert systems", labels)
        self.assertIn("jakarta", labels)
        self.assertFalse(any(label.startswith("the ") for label in labels))


if __name__ == "__main__":
    unittest.main()
