"""Regression coverage for Plex caching, provenance, playlist controls and PCM joins.

Plex and Genius requests are mocked. Audio integration tests use tiny synthetic
files and the real FFmpeg executable; no server, tokens, model downloads or GPU.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
from mutagen.id3 import ID3, USLT

from aimtv.audio_ops import _hard_concat_wavs, duration, stitch_like_airadio
from aimtv.context import resolve_contexts
from aimtv.playlist import Clip, PlexConfig, build_review_playlist, pick_interstitials
from aimtv.plex_playlist import PlexPlaylistSource, _extract_embedded_lyrics, _write_text_atomic


def make_source(root: Path, server_id: str = "server-a", section_type: str = "artist"):
    server = mock.Mock()
    server.machineIdentifier = server_id
    server.library.section.return_value.type = section_type
    with mock.patch("aimtv.plex_playlist.PlexServer", return_value=server):
        source = PlexPlaylistSource("http://plex:32400", "test-token", "Music", cache_dir=root)
    return source


@pytest.fixture
def track(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"placeholder")
    return SimpleNamespace(
        ratingKey="17", title="Test Song", grandparentTitle="Test Artist",
        parentTitle="Test Album", locations=[str(audio)],
    )


def test_cache_is_isolated_between_servers(tmp_path, track):
    first = make_source(tmp_path / "cache", "server-one")
    second = make_source(tmp_path / "cache", "server-two")
    audio = Path(track.locations[0])
    with mock.patch("aimtv.plex_playlist._extract_embedded_lyrics", side_effect=["first lyrics", "second lyrics"]):
        left, _ = first._lyrics_for(track, audio)
        right, _ = second._lyrics_for(track, audio)
    assert left != right
    assert left.read_text() == "first lyrics"
    assert right.read_text() == "second lyrics"
    assert "test-token" not in str(left)
    assert first.server_id == hashlib.sha256(b"server-one").hexdigest()


def test_cache_hit_avoids_repeated_lyric_extraction(tmp_path, track):
    source = make_source(tmp_path / "cache")
    with mock.patch("aimtv.plex_playlist._extract_embedded_lyrics", return_value="exact lyrics\n") as extract:
        first = source._lyrics_for(track, Path(track.locations[0]))
        second = source._lyrics_for(track, Path(track.locations[0]))
    assert first == second
    assert first[1] == "embedded"
    extract.assert_called_once()


def test_empty_cache_is_retried_when_genius_becomes_available(tmp_path, track):
    source = make_source(tmp_path / "cache")
    audio = Path(track.locations[0])
    with mock.patch("aimtv.plex_playlist._extract_embedded_lyrics", return_value=None):
        cache, origin = source._lyrics_for(track, audio)
        assert cache.read_text() == ""
        assert origin is None
        source.genius = mock.Mock()
        source.genius.search_song.return_value = SimpleNamespace(lyrics="recovered lyrics")
        cache, origin = source._lyrics_for(track, audio)
    assert cache.read_text() == "recovered lyrics"
    assert origin == "genius"


def test_transient_genius_failure_does_not_poison_cache(tmp_path, track):
    source = make_source(tmp_path / "cache")
    source.genius = mock.Mock()
    source.genius.search_song.side_effect = [TimeoutError(), SimpleNamespace(lyrics="retry succeeded")]
    with mock.patch("aimtv.plex_playlist._extract_embedded_lyrics", return_value=None):
        assert source._lyrics_for(track, Path(track.locations[0]))[1] is None
        cache, origin = source._lyrics_for(track, Path(track.locations[0]))
    assert origin == "genius"
    assert cache.read_text() == "retry succeeded"


def test_embedded_lyrics_take_priority_over_genius(tmp_path, track):
    source = make_source(tmp_path / "cache")
    source.genius = mock.Mock()
    with mock.patch("aimtv.plex_playlist._extract_embedded_lyrics", return_value="embedded text"):
        assert source._lyrics_for(track, Path(track.locations[0]))[1] == "embedded"
    source.genius.search_song.assert_not_called()


@pytest.mark.parametrize("change", ["audio", "title", "artist", "album", "text", "marker", "missing-marker", "invalid-utf8"])
def test_changed_or_corrupt_cache_is_rebuilt(tmp_path, track, change):
    source = make_source(tmp_path / "cache")
    audio = Path(track.locations[0])
    with mock.patch("aimtv.plex_playlist._extract_embedded_lyrics", side_effect=["original", "refreshed"]) as extract:
        cache, _ = source._lyrics_for(track, audio)
        marker = cache.with_suffix(".json")
        if change == "audio":
            audio.write_bytes(b"a changed file with updated tags")
        elif change == "title":
            track.title = "New Title"
        elif change == "artist":
            track.grandparentTitle = "New Artist"
        elif change == "album":
            track.parentTitle = "New Album"
        elif change == "text":
            cache.write_text("corrupted text")
        elif change == "marker":
            marker.write_text("[]")
        elif change == "missing-marker":
            marker.unlink()
        elif change == "invalid-utf8":
            cache.write_bytes(b"\xff\xfe")
        cache, origin = source._lyrics_for(track, audio)
    assert extract.call_count == 2
    assert cache.read_text() == "refreshed"
    assert origin == "embedded"


def test_newline_normalization_keeps_cache_hash_consistent(tmp_path, track):
    source = make_source(tmp_path / "cache")
    with mock.patch("aimtv.plex_playlist._extract_embedded_lyrics", return_value="one\r\ntwo\rthree") as extract:
        cache, _ = source._lyrics_for(track, Path(track.locations[0]))
        source._lyrics_for(track, Path(track.locations[0]))
    assert cache.read_bytes() == b"one\ntwo\nthree"
    saved = json.loads(cache.with_suffix(".json").read_text())
    assert saved["lyrics_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    extract.assert_called_once()


def test_cache_write_failure_preserves_existing_file_and_cleans_temp(tmp_path):
    cache = tmp_path / "lyrics.txt"
    cache.write_text("original")
    with mock.patch.object(Path, "replace", side_effect=OSError("cannot replace")):
        with pytest.raises(OSError):
            _write_text_atomic(cache, "replacement")
    assert cache.read_text() == "original"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["lyrics.txt"]


def test_non_music_section_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not a music library"):
        make_source(tmp_path, section_type="movie")


def test_track_limits_do_not_scan_for_zero(tmp_path):
    source = make_source(tmp_path)
    assert source.track_keys(limit=0) == []
    source.section.searchTracks.assert_not_called()
    with pytest.raises(ValueError, match="non-negative"):
        source.track_keys(limit=-1)
    source.section.searchTracks.return_value = [SimpleNamespace(ratingKey="7")]
    assert source.track_keys(limit=5) == [7]
    source.section.searchTracks.assert_called_once_with(maxresults=5)


def test_local_file_search_checks_all_locations(tmp_path, track):
    actual = Path(track.locations[0])
    track.locations.insert(0, str(tmp_path / "missing.mp3"))
    assert PlexPlaylistSource._local_file_path(track) == actual
    actual.unlink()
    with pytest.raises(RuntimeError, match="not readable locally"):
        PlexPlaylistSource._local_file_path(track)


def test_resolved_metadata_includes_server_identity(tmp_path, track):
    source = make_source(tmp_path / "cache")
    with mock.patch("aimtv.plex_playlist._extract_embedded_lyrics", return_value="lyrics"):
        _, _, metadata = source._resolve(track)
    assert metadata["server_id"] == source.server_id
    assert metadata["rating_key"] == 17


def test_id3_lyrics_are_read_from_real_tags(tmp_path):
    path = tmp_path / "tag-only.mp3"
    tags = ID3()
    tags.add(USLT(encoding=3, lang="eng", desc="", text="tagged lyrics"))
    tags.save(path)
    assert _extract_embedded_lyrics(path) == "tagged lyrics"


@pytest.mark.parametrize("key", ["LYRICS", "UNSYNCEDLYRICS", "lyrics", "\u00a9lyr"])
def test_mapping_style_lyrics_tags(tmp_path, key):
    with mock.patch("mutagen.File", return_value=SimpleNamespace(tags={key: ["tagged lyrics"]})):
        assert _extract_embedded_lyrics(tmp_path / "track") == "tagged lyrics"


def test_unknown_lyric_source_is_not_verified(tmp_path):
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("nonempty but untraceable")
    clip = Clip(tmp_path / "track.mp3", "song", "Track", lyrics_path=lyrics)
    context = resolve_contexts([clip], tmp_path)[0]
    assert not context.verified
    assert "source" in context.warning


def test_invalid_utf8_lyrics_are_unverified_not_a_crash(tmp_path):
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_bytes(b"\xff\xfe")
    clip = Clip(tmp_path / "track.mp3", "song", "Track", lyrics_path=lyrics, lyrics_source="embedded")
    context = resolve_contexts([clip], tmp_path)[0]
    assert not context.verified
    assert "unreadable" in context.warning


def test_connection_repr_does_not_expose_tokens():
    config = PlexConfig("http://plex", "secret-plex", "Music", "secret-genius")
    assert "secret-plex" not in repr(config)
    assert "secret-genius" not in repr(config)
    assert config.token == "secret-plex"


def test_zero_interstitials_does_not_scan_or_insert(tmp_path):
    with mock.patch("aimtv.playlist.count_interstitials") as scan:
        assert pick_interstitials(tmp_path, 0, random.Random(1)) == []
    scan.assert_not_called()
    songs = [Clip(tmp_path / f"song-{i}.wav", "song", str(i)) for i in range(2)]
    with mock.patch("aimtv.playlist.get_airadio_song_clips", return_value=songs):
        assert build_review_playlist(interstitial_min=0, interstitial_max=0, seed=1) == songs


def test_interstitial_sampling_is_independent_of_filesystem_order(tmp_path):
    paths = [tmp_path / f"ad-{i}.wav" for i in range(6)]
    with mock.patch("aimtv.playlist.count_interstitials", side_effect=[paths, []]):
        first = pick_interstitials(tmp_path, 3, random.Random(42))
    with mock.patch("aimtv.playlist.count_interstitials", side_effect=[list(reversed(paths)), []]):
        second = pick_interstitials(tmp_path, 3, random.Random(42))
    assert first == second


ffmpeg_available = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg/ffprobe required",
)


def write_tone(path: Path, rate: int, channels: int, seconds: float = 0.4, subtype: str = "PCM_16"):
    tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(round(rate * seconds)) / rate)
    sf.write(path, np.tile(tone[:, None], (1, channels)), rate, subtype=subtype)
    return path


@ffmpeg_available
@pytest.mark.parametrize("second_rate,second_channels,subtype", [(48000, 1, "PCM_16"), (44100, 1, "PCM_16"), (44100, 2, "PCM_24")])
def test_mixed_pcm_join_preserves_duration_and_pitch(tmp_path, second_rate, second_channels, subtype):
    first = write_tone(tmp_path / "first.wav", 44100, 2)
    second = write_tone(tmp_path / "second.wav", second_rate, second_channels, subtype=subtype)
    output = _hard_concat_wavs([first, second], tmp_path / "joined.wav")
    data, rate = sf.read(output, always_2d=True)
    assert rate == 44100
    assert data.shape[1] == 2
    assert len(data) / rate == pytest.approx(0.8, abs=0.002)
    for start in (0.05, 0.45):
        segment = data[round(start * rate):round((start + 0.3) * rate), 0]
        frequency = np.fft.rfftfreq(len(segment), 1 / rate)[np.argmax(np.abs(np.fft.rfft(segment)))]
        assert frequency == pytest.approx(440, abs=4)


@ffmpeg_available
def test_matching_pcm_join_preserves_samples_exactly(tmp_path):
    first = write_tone(tmp_path / "first.wav", 44100, 2)
    second = write_tone(tmp_path / "second.wav", 44100, 2)
    expected = np.concatenate([sf.read(p, dtype="int16", always_2d=True)[0] for p in (first, second)])
    output = _hard_concat_wavs([first, second], tmp_path / "joined.wav")
    actual, _ = sf.read(output, dtype="int16", always_2d=True)
    np.testing.assert_array_equal(actual, expected)


@ffmpeg_available
def test_mixed_song_and_interstitial_timeline_stays_in_sync(tmp_path):
    specs = [(44100, 2, "song"), (22050, 1, "ad"), (48000, 2, "song")]
    clips = [Clip(write_tone(tmp_path / f"clip-{i}.wav", rate, channels, seconds=3), kind, str(i))
             for i, (rate, channels, kind) in enumerate(specs)]
    output, starts, durations = stitch_like_airadio(clips, tmp_path / "work", tmp_path / "timeline.wav")
    assert starts == pytest.approx([0, 3, 3.5], abs=0.002)
    assert durations == pytest.approx([3, 3, 3], abs=0.002)
    assert duration(output) == pytest.approx(6.5, abs=0.003)


def test_single_piece_copy_does_not_load_whole_file(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"example")
    with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file read")):
        output = _hard_concat_wavs([source], tmp_path / "output.wav")
    assert output.read_bytes() == b"example"


def test_atomic_writer_can_create_new_cache_file(tmp_path):
    with tempfile.TemporaryDirectory(dir=tmp_path) as raw:
        path = Path(raw) / "lyrics.txt"
        _write_text_atomic(path, "complete lyrics\n")
        assert path.read_text() == "complete lyrics\n"
        assert list(path.parent.iterdir()) == [path]
