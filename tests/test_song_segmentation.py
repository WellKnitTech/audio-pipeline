from __future__ import annotations

import unittest

from audio_pipeline.song_segmentation import _aligned_chunk_samples, _audio_start_offset, _ffmpeg_seek_seconds, _video_bitrate, _video_timeline, applause_spans, validate_cuts


class ApplauseSpanTests(unittest.TestCase):
    def test_scanner_chunks_align_to_the_yamnet_frame_grid(self) -> None:
        hop_samples = 16_000 * 48 // 100
        chunk_samples = _aligned_chunk_samples(30)
        self.assertEqual(chunk_samples % hop_samples, 0)
        self.assertGreaterEqual(chunk_samples, 30 * 16_000)

    def test_video_cut_timeline_uses_stream_origin_and_duration(self) -> None:
        source_info = {
            "duration_seconds": 6.0,
            "streams": [{"codec_type": "video", "start_time": "1.0", "duration": "5.0"}],
        }
        self.assertEqual(_video_timeline(source_info), (1.0, 5.0))
        source_info["streams"][0].pop("duration")
        source_info["format_start_time_seconds"] = 0.956
        source_info["duration_seconds"] = 5.044
        self.assertEqual(_video_timeline(source_info), (1.0, 5.0))

    def test_ffmpeg_seek_maps_video_relative_time_to_container_timeline(self) -> None:
        source_info = {
            "duration_seconds": 6.0,
            "format_start_time_seconds": 0.0,
            "streams": [{"codec_type": "video", "start_time": "1.0", "duration": "5.0"}],
        }
        self.assertEqual(_ffmpeg_seek_seconds(source_info, 1.0), 2.0)
        source_info["format_start_time_seconds"] = 0.978
        self.assertEqual(_ffmpeg_seek_seconds(source_info, 1.0), 1.022)

    def test_audio_offset_is_relative_to_first_video_stream(self) -> None:
        streams = [
            {"codec_type": "video", "start_time": "0.5"},
            {"codec_type": "audio", "start_time": "2.5"},
        ]
        self.assertEqual(_audio_start_offset(streams), 2.0)
        streams[1]["start_time"] = "0.25"
        self.assertEqual(_audio_start_offset(streams), -0.25)

    def test_merges_sustained_applause_but_rejects_choir_clapping(self) -> None:
        rows = [
            {"time_seconds": 0.0, "applause": 0.0, "clapping": 0.8, "cheering": 0.0, "singing": 0.0, "choir": 0.0},
            {"time_seconds": 0.48, "applause": 0.0, "clapping": 0.7, "cheering": 0.0, "singing": 0.7, "choir": 0.8},
            {"time_seconds": 10.0, "applause": 0.6, "clapping": 0.0, "cheering": 0.0, "singing": 0.0, "choir": 0.0},
            {"time_seconds": 10.48, "applause": 0.7, "clapping": 0.0, "cheering": 0.0, "singing": 0.0, "choir": 0.0},
            {"time_seconds": 11.44, "applause": 0.8, "clapping": 0.0, "cheering": 0.0, "singing": 0.0, "choir": 0.0},
            {"time_seconds": 11.92, "applause": 0.7, "clapping": 0.0, "cheering": 0.0, "singing": 0.0, "choir": 0.0},
        ]
        self.assertEqual(applause_spans(rows, threshold=0.5, merge_gap_seconds=1.0, min_duration_seconds=1.5), [
            {"start_seconds": 10.0, "end_seconds": 12.4, "peak_score": 0.8}
        ])

    def test_bitrate_target_decreases_for_longer_clips_and_rejects_impossible_size(self) -> None:
        short = _video_bitrate(60, 95, 128_000)
        long = _video_bitrate(600, 95, 128_000)
        self.assertGreater(short, long)
        with self.assertRaisesRegex(ValueError, "too small"):
            _video_bitrate(3600, 1, 128_000)

    def test_validates_cut_ranges_and_rejects_overlaps(self) -> None:
        valid = [
            {"label": "Song 1", "start_seconds": 10, "end_seconds": 20},
            {"label": "Song 2", "start_seconds": 25, "end_seconds": 35},
        ]
        self.assertEqual(validate_cuts(valid, source_duration=40), valid)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_cuts([
                {"label": "Song 1", "start_seconds": 10, "end_seconds": 20},
                {"label": "Song 2", "start_seconds": 19, "end_seconds": 30},
            ], source_duration=40)
        with self.assertRaisesRegex(ValueError, "source duration"):
            validate_cuts([{"label": "Song 1", "start_seconds": 10, "end_seconds": 40.01}], source_duration=40)

if __name__ == "__main__":
    unittest.main()
