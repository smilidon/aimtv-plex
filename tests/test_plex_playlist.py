from __future__ import annotations

import hashlib
import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aimtv.cli import _playlist_kwargs, build_parser
from aimtv.context import resolve_contexts
from aimtv.playlist import Clip, PlexConfig, get_plex_song_clips


class _FakeSource:
    """Stands in for PlexPlaylistSource: three tracks, one without lyrics."""

    instances: list["_FakeSource"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.resolved: list[int] = []
        root = Path(kwargs["cache_dir"])
        root.mkdir(parents=True, exist_ok=True)
        self.tracks = {}
        for key, title, lyrics, source in (
            (11, "Neon Hedgehog", "neon lights\nhedgehog nights\n", "embedded"),
            (12, "Copper Systems", "copper wire\ncopper fire\n", "genius"),
            (13, "Silent Track", "", None),
        ):
            audio = root / f"{key}.mp3"
            audio.write_bytes(b"mp3-placeholder")
            text = root / f"{key}.lyrics.txt"
            text.write_text(lyrics, encoding="utf-8")
            self.tracks[key] = (audio, text, {"rating_key": key, "title": title, "lyrics_source": source})
        _FakeSource.instances.append(self)

    def track_keys(self, *, limit=None):
        return sorted(self.tracks)[:limit]

    def resolve_tracks(self, keys):
        for key in keys:
            self.resolved.append(key)
            yield self.tracks[key]


class PlexPlaylistTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeSource.instances.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name)
        self.env = mock.patch.dict(os.environ, {"AIMTV_CACHE_HOME": self._tmp.name})
        self.env.start()
        self.source = mock.patch("aimtv.plex_playlist.PlexPlaylistSource", _FakeSource)
        self.source.start()
        self.config = PlexConfig(url="http://plex:32400", token="tok", section="Music")

    def tearDown(self) -> None:
        self.source.stop()
        self.env.stop()
        self._tmp.cleanup()

    def test_samples_deterministically_and_resolves_only_chosen_tracks(self) -> None:
        first = get_plex_song_clips(self.config, song_count=2, seed=7)
        second = get_plex_song_clips(self.config, song_count=2, seed=7)
        self.assertEqual([c.title for c in first], [c.title for c in second])
        self.assertEqual(len(first), 2)
        self.assertEqual(len(_FakeSource.instances[-1].resolved), 2)
        for clip in first:
            self.assertEqual(clip.kind, "song")
            self.assertTrue(clip.provenance_id.startswith("plex:"))
            self.assertIsNotNone(clip.lyrics_path)

    def test_passes_connection_details_to_source(self) -> None:
        get_plex_song_clips(PlexConfig("http://h", "t", "Music", genius_token="g"), song_count=1, seed=1)
        kwargs = _FakeSource.instances[-1].kwargs
        self.assertEqual(kwargs["baseurl"], "http://h")
        self.assertEqual(kwargs["token"], "t")
        self.assertEqual(kwargs["section_name"], "Music")
        self.assertEqual(kwargs["genius_token"], "g")
        self.assertEqual(Path(kwargs["cache_dir"]), self.cache / "plex")

    def test_refuses_when_section_is_too_small(self) -> None:
        with self.assertRaises(RuntimeError):
            get_plex_song_clips(self.config, song_count=4, seed=1)


class PlexContextTests(unittest.TestCase):
    def test_sourced_lyrics_are_hashed_and_traceable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            audio = root / "a.mp3"
            audio.write_bytes(b"x")
            text = root / "a.txt"
            text.write_text("hello\nworld\n", encoding="utf-8")
            clip = Clip(audio, "song", "A", lyrics_path=text, lyrics_source="embedded", provenance_id="plex:11")
            context = resolve_contexts([clip], root)[0]
            self.assertTrue(context.verified)
            self.assertEqual(context.lyrics, "hello\nworld\n")
            digest = hashlib.sha256(b"hello\nworld\n").hexdigest()
            self.assertEqual(context.lyrics_sha256, digest)
            self.assertEqual(context.source, "plex embedded lyrics")
            self.assertTrue(context.provenance_id.startswith("plex:11:embedded:"))

    def test_empty_lyrics_are_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            audio = root / "a.mp3"
            audio.write_bytes(b"x")
            text = root / "a.txt"
            text.write_text("", encoding="utf-8")
            clip = Clip(audio, "song", "A", lyrics_path=text, provenance_id="plex:13")
            context = resolve_contexts([clip], root)[0]
            self.assertFalse(context.verified)
            self.assertIsNone(context.lyrics)
            self.assertIn("no embedded lyrics", context.warning)


class PlexCliTests(unittest.TestCase):
    def test_playlist_kwargs_match_plan_and_render_signatures(self) -> None:
        from aimtv.planning import build_review_plan
        from aimtv.review import render_review_mp4

        args = build_parser().parse_args(["plan"])
        keys = set(_playlist_kwargs(args))
        for fn in (build_review_plan, render_review_mp4):
            self.assertLessEqual(keys, set(inspect.signature(fn).parameters), fn.__name__)

    def test_no_plex_flags_means_no_plex_config(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            args = build_parser().parse_args(["plan"])
            self.assertIsNone(_playlist_kwargs(args)["plex"])

    def test_flags_and_env_combine(self) -> None:
        with mock.patch.dict(os.environ, {"PLEX_TOKEN": "env-token", "GENIUS_TOKEN": "g"}, clear=True):
            args = build_parser().parse_args(["plan", "--plex-url", "http://h", "--plex-section", "Music"])
            plex = _playlist_kwargs(args)["plex"]
        self.assertEqual(plex, PlexConfig("http://h", "env-token", "Music", genius_token="g"))

    def test_partial_plex_config_is_rejected(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            args = build_parser().parse_args(["plan", "--plex-url", "http://h"])
            with self.assertRaises(SystemExit) as ctx:
                _playlist_kwargs(args)
        self.assertIn("PLEX_TOKEN", str(ctx.exception))
        self.assertIn("PLEX_SECTION", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
