from __future__ import annotations

import unittest
from pathlib import Path

from aimtv.alignment import AlignmentResult, UnitAlignment
from aimtv.context import ClipContext
from aimtv.playlist import Clip
from aimtv.prompts import PromptBank
from aimtv.schedule import build_prompt_schedule


class ScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clips = [
            Clip(Path("one.wav"), "song", "Neon Hedgehog"),
            Clip(Path("ad.wav"), "ad", "Jakarta Ad"),
            Clip(Path("two.wav"), "song", "Copper Systems"),
        ]
        self.contexts = [
            ClipContext(
                Path("one.wav"),
                "song",
                "Neon Hedgehog",
                "[Verse]\nA neon hedgehog crosses Jakarta\n[Chorus]\nCyberpunk rain returns",
                "a" * 64,
                "song-lyrics:1:aaaaaaaaaaaaaaaa",
                True,
                "test",
            ),
            ClipContext(
                Path("ad.wav"),
                "ad",
                "Jakarta Ad",
                "Expert systems recommend crab milk. Pandu declines politely.",
                "b" * 64,
                "interstitial-1",
                True,
                "test",
            ),
            ClipContext(
                Path("two.wav"),
                "song",
                "Copper Systems",
                "[Verse]\nCopper systems wake\n[Chorus]\nWassies orbit Indonesia",
                "c" * 64,
                "song-lyrics:2:cccccccccccccccc",
                True,
                "test",
            ),
        ]
        self.bank = PromptBank(
            prompts=("public access transmission",),
            subjects=("hedgey hog",),
            style_suffix="vivid analog television",
            source=Path("prompts.json"),
        )

    def test_context_covers_crossfade_timeline_and_tracks_source(self) -> None:
        schedule = build_prompt_schedule(
            self.clips,
            self.contexts,
            starts=[0.0, 20.0, 27.5],
            durs=[20.0, 10.0, 20.0],
            bank=self.bank,
            seed=99,
            min_random_gap_s=100.0,
            max_random_gap_s=100.0,
        )
        self.assertEqual(schedule.contextual[0].start_s, 0.0)
        self.assertEqual(schedule.contextual[-1].end_s, 47.5)
        # Ad/song overlap is split at its midpoint: 27.5 + (30 - 27.5)/2.
        boundary = 28.75
        ad_events = [e for e in schedule.contextual if e.clip_index == 1]
        song_events = [e for e in schedule.contextual if e.clip_index == 2]
        self.assertEqual(ad_events[-1].end_s, boundary)
        self.assertEqual(song_events[0].start_s, boundary)
        self.assertTrue(all(event.verified for event in schedule.contextual))
        self.assertTrue(all(event.source_text for event in schedule.contextual))
        self.assertTrue(all(event.provenance_id for event in schedule.contextual))
        self.assertTrue(all("vivid analog television" in event.prompt for event in schedule.contextual))

    def test_random_cutaways_are_deterministic_and_bounded(self) -> None:
        kwargs = dict(
            clips=self.clips,
            contexts=self.contexts,
            starts=[0.0, 60.0, 70.0],
            durs=[60.0, 12.0, 70.0],
            bank=self.bank,
            seed=11,
            min_random_gap_s=10.0,
            max_random_gap_s=10.0,
            min_random_duration_s=5.0,
            max_random_duration_s=5.0,
        )
        first = build_prompt_schedule(**kwargs)
        second = build_prompt_schedule(**kwargs)
        self.assertEqual(first, second)
        self.assertGreater(len(first.random_cutaways), 0)
        for event in first.random_cutaways:
            self.assertEqual(event.end_s - event.start_s, 5.0)
            self.assertEqual(event.event_type, "random-cutaway")
            self.assertTrue(event.hard_reset)
            self.assertIn("hedgey hog", event.prompt)
            # Dream footage is allowed during songs, never over an interstitial.
            self.assertFalse(event.start_s < 71.0 and event.end_s > 60.0)

    def test_voice_anchor_timing_is_used_and_protected_from_dream_cuts(self) -> None:
        alignment = AlignmentResult(
            schema_version=1,
            audio_sha256="d" * 64,
            lyrics_sha256="a" * 64,
            model_id="test-local-asr",
            recognized_text="cyberpunk rain returns",
            matched_token_ratio=0.5,
            units=(
                UnitAlignment(0, None, None, 0.0, (), "unmatched"),
                UnitAlignment(
                    1,
                    30.0,
                    32.0,
                    0.95,
                    ("Cyberpunk", "rain", "returns"),
                    "voice-asr-monotonic",
                ),
            ),
            cache_path=Path("alignment.json"),
        )
        schedule = build_prompt_schedule(
            self.clips,
            self.contexts,
            starts=[0.0, 60.0, 70.0],
            durs=[60.0, 12.0, 70.0],
            bank=self.bank,
            seed=13,
            min_random_gap_s=1.0,
            max_random_gap_s=1.0,
            min_random_duration_s=5.0,
            max_random_duration_s=5.0,
            alignments={str(self.clips[0].path.resolve()): alignment},
        )
        anchored = next(
            event
            for event in schedule.contextual
            if event.clip_index == 0 and event.source_text == "Cyberpunk rain returns"
        )
        self.assertEqual(anchored.start_s, 29.25)
        self.assertEqual(anchored.recognized_start_s, 30.0)
        self.assertEqual(anchored.alignment_method, "voice-asr-monotonic")
        self.assertFalse(
            any(
                event.start_s < 34.0 and event.end_s > 29.25
                for event in schedule.random_cutaways
            )
        )

    def test_interstitial_prompts_only_use_verified_script_phrases(self) -> None:
        schedule = build_prompt_schedule(
            self.clips,
            self.contexts,
            starts=[0.0, 20.0, 30.0],
            durs=[20.0, 10.0, 20.0],
            bank=self.bank,
            seed=17,
            min_random_gap_s=1.0,
            max_random_gap_s=1.0,
            min_random_duration_s=5.0,
            max_random_duration_s=5.0,
        )
        script_words = set(
            "expert systems recommend crab milk pandu declines politely".split()
        )
        interstitial_events = [
            event for event in schedule.contextual if event.clip_index == 1
        ]
        self.assertGreater(len(interstitial_events), 0)
        for event in interstitial_events:
            for phrase in event.keyphrases:
                self.assertTrue(set(phrase.lower().split()) <= script_words)
        interstitial_start = min(event.start_s for event in interstitial_events)
        interstitial_end = max(event.end_s for event in interstitial_events)
        self.assertFalse(
            any(
                event.start_s < interstitial_end and event.end_s > interstitial_start
                for event in schedule.random_cutaways
            )
        )


if __name__ == "__main__":
    unittest.main()
