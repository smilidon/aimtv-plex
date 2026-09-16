from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from airadio import interstitial_provenance, lyricist

from aimtv.context import resolve_contexts
from aimtv.playlist import Clip


class ContextResolutionTests(unittest.TestCase):
    def test_song_lyrics_are_reconstructed_and_hash_verified(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            home = Path(raw)
            song = home / "library" / "verified-song.wav"
            song.parent.mkdir(parents=True)
            song.write_bytes(b"wav-placeholder")
            title = "Hedgehog Skyline"
            lyric_id = 7
            lyrics = lyricist.compose_lyrics(title, lyric_id)
            (home / "library" / "catalog.json").write_text(
                json.dumps(
                    {
                        "songs": [
                            {
                                "path": "library/verified-song.wav",
                                "title": title,
                                "lyric_id": lyric_id,
                                "lyrics_sha256": lyricist.lyrics_sha256(lyrics),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            context = resolve_contexts([Clip(song, "song", title)], home)[0]
            self.assertTrue(context.verified)
            self.assertEqual(context.lyrics, lyrics)
            self.assertEqual(context.lyrics_sha256, lyricist.lyrics_sha256(lyrics))
            self.assertIn(f"song-lyrics:{lyric_id}:", context.provenance_id or "")

    def test_song_hash_mismatch_is_explicitly_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            home = Path(raw)
            song = home / "library" / "bad.wav"
            song.parent.mkdir(parents=True)
            song.write_bytes(b"wav-placeholder")
            (home / "library" / "catalog.json").write_text(
                json.dumps(
                    {
                        "songs": [
                            {
                                "path": "library/bad.wav",
                                "title": "Bad Hash",
                                "lyric_id": 2,
                                "lyrics_sha256": "0" * 64,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            context = resolve_contexts([Clip(song, "song", "Bad Hash")], home)[0]
            self.assertFalse(context.verified)
            self.assertIn("failed hash verification", context.warning or "")

    def test_interstitial_reads_provenance_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            home = Path(raw)
            audio = home / "interstitials" / "audio" / "ads" / "ad.wav"
            audio.parent.mkdir(parents=True)
            audio.write_bytes(b"wav-placeholder")
            lyrics = "Try cyberpunk crab milk. Broadcast from Jakarta."
            record = interstitial_provenance.record_generation(
                home,
                audio,
                lyrics=lyrics,
                kind="ad",
                style="test",
                backend="test",
            )

            context = resolve_contexts([Clip(audio, "ad", "Test Ad")], home)[0]
            self.assertTrue(context.verified)
            self.assertEqual(context.lyrics, lyrics)
            self.assertEqual(context.provenance_id, record["generation_id"])
            self.assertEqual(context.lyrics_sha256, record["lyrics"]["sha256"])


if __name__ == "__main__":
    unittest.main()
