from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_pipeline import video


class VideoPreparationTests(unittest.TestCase):
    def test_extracts_bounded_audio_chunks_and_writes_timeline_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "event.mp4"
            source.write_bytes(b"original video bytes")
            output = root / "prepared"
            calls: list[list[str]] = []

            def fake_run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
                calls.append(cmd)
                if cmd[0] == "ffmpeg":
                    pattern = Path(cmd[-1])
                    pattern.parent.mkdir(parents=True, exist_ok=True)
                    (pattern.parent / "chunk_0000.wav").write_bytes(b"chunk 1")
                    (pattern.parent / "chunk_0001.wav").write_bytes(b"chunk 2")
                    return subprocess.CompletedProcess(cmd, 0, "")
                if cmd[-1] == str(source):
                    payload = {"format": {"duration": "12.5", "size": str(source.stat().st_size)},
                               "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080, "start_time": "0.000000"},
                                           {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2, "start_time": "2.000000"}]}
                else:
                    payload = {"format": {"duration": "10.0" if "0000" in cmd[-1] else "2.5"}, "streams": []}
                return subprocess.CompletedProcess(cmd, 0, json.dumps(payload))

            with patch.object(video, "run_cmd", side_effect=fake_run):
                result = video.prepare_video(source, output, chunk_seconds=10)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(result["duration_seconds"], 12.5)
            self.assertEqual([(c["start_seconds"], c["duration_seconds"]) for c in manifest["chunks"]],
                             [(2.0, 10.0), (12.0, 2.5)])
            self.assertTrue((output / "audio" / "chunk_0000.wav").is_file())
            self.assertEqual(source.read_bytes(), b"original video bytes")
            ffmpeg_call = next(cmd for cmd in calls if cmd[0] == "ffmpeg")
            self.assertIn("-segment_time", ffmpeg_call)
            self.assertIn("10", ffmpeg_call)

    def test_sorts_segment_numbers_numerically_after_four_digits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "event.mp4"
            source.write_bytes(b"video")
            output = root / "prepared"

            def fake_run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
                if cmd[0] == "ffmpeg":
                    pattern = Path(cmd[-1])
                    for index in (1000, 10000, 1001):
                        (pattern.parent / f"chunk_{index:04d}.wav").touch()
                    return subprocess.CompletedProcess(cmd, 0, "")
                if cmd[-1] == str(source):
                    payload = {"format": {"duration": "3.0"},
                               "streams": [{"codec_type": "video", "start_time": "0"},
                                           {"codec_type": "audio", "start_time": "0"}]}
                else:
                    payload = {"format": {"duration": "1.0"}, "streams": []}
                return subprocess.CompletedProcess(cmd, 0, json.dumps(payload))

            with patch.object(video, "run_cmd", side_effect=fake_run):
                manifest = video.prepare_video(source, output, chunk_seconds=1)

            self.assertEqual([Path(c["file"]).name for c in manifest["chunks"]],
                             ["chunk_1000.wav", "chunk_1001.wav", "chunk_10000.wav"])

    def test_refuses_to_overwrite_an_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "event.mp4"
            source.write_bytes(b"video")
            output = root / "prepared"
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                video.prepare_video(source, output, chunk_seconds=10)
            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
