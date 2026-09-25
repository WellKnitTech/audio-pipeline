from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audio_pipeline.video import merge_chunk_transcripts


class VideoTranscriptMergeTests(unittest.TestCase):
    def test_rebases_and_orders_segments_from_chunk_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            transcript_dir = root / "transcripts"
            transcript_dir.mkdir()
            output_path = root / "event.json"
            manifest_path.write_text(json.dumps({
                "source_file": "/media/event.mp4",
                "duration_seconds": 80.0,
                "timeline_origin": "first_video_stream_start",
                "chunks": [
                    {"file": "audio/chunk_0001.wav", "start_seconds": 60.0, "duration_seconds": 20.0},
                    {"file": "audio/chunk_0000.wav", "start_seconds": 0.0, "duration_seconds": 60.0},
                ],
            }), encoding="utf-8")
            (transcript_dir / "chunk_0000.json").write_text(json.dumps({"segments": [
                {"start": 4.0, "end": 5.5, "text": "First chunk."},
            ]}), encoding="utf-8")
            (transcript_dir / "chunk_0001.asr16k.json").write_text(json.dumps({"segments": [
                {"start": 1.0, "end": 2.0, "text": "Second chunk."},
            ]}), encoding="utf-8")

            result = merge_chunk_transcripts(manifest_path, transcript_dir, output_path)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual([(s["start"], s["end"], s["text"]) for s in saved["segments"]], [
                (4.0, 5.5, "First chunk."),
                (61.0, 62.0, "Second chunk."),
            ])
            self.assertEqual([s["chunk"] for s in saved["segments"]], ["chunk_0000.wav", "chunk_0001.wav"])
            self.assertEqual(saved["meta"]["source_file"], "/media/event.mp4")
            self.assertEqual(result, saved)

    def test_rejects_non_finite_rebased_times_without_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            transcript_dir = root / "transcripts"
            transcript_dir.mkdir()
            output_path = root / "event.json"
            manifest_path.write_text(json.dumps({"chunks": [
                {"file": "audio/chunk_0000.wav", "start_seconds": 1e308},
            ]}), encoding="utf-8")
            (transcript_dir / "chunk_0000.json").write_text(json.dumps({"segments": [
                {"start": 1e308, "end": 1e308, "text": "overflow"},
            ]}), encoding="utf-8")

            with self.assertRaises(ValueError):
                merge_chunk_transcripts(manifest_path, transcript_dir, output_path)
            self.assertFalse(output_path.exists())

    def test_missing_chunk_transcript_fails_without_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            transcript_dir = root / "transcripts"
            transcript_dir.mkdir()
            output_path = root / "event.json"
            manifest_path.write_text(json.dumps({"chunks": [
                {"file": "audio/chunk_0000.wav", "start_seconds": 0.0},
            ]}), encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                merge_chunk_transcripts(manifest_path, transcript_dir, output_path)
            self.assertFalse(output_path.exists())

    def test_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            transcript_dir = root / "transcripts"
            transcript_dir.mkdir()
            output_path = root / "event.json"
            output_path.write_text("keep", encoding="utf-8")
            manifest_path.write_text(json.dumps({"chunks": []}), encoding="utf-8")

            with self.assertRaises(FileExistsError):
                merge_chunk_transcripts(manifest_path, transcript_dir, output_path)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
