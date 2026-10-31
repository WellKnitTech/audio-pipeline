from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

from .video import probe_media
from .utils import run_cmd

SAMPLE_RATE = 16_000
HOP_SECONDS = 0.48
CLASS_IDS = {"applause": 62, "clapping": 58, "cheering": 61, "music": 132, "singing": 24, "choir": 25}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def applause_spans(
    rows: Iterable[dict[str, Any]], *, threshold: float = 0.18,
    merge_gap_seconds: float = 3.0, min_duration_seconds: float = 1.5,
    vocal_ceiling: float = 0.25, hop_seconds: float = HOP_SECONDS,
) -> list[dict[str, float]]:
    """Find candidate audience-response spans; exclude frames still classed as singing/choir."""
    threshold = _finite(threshold, "threshold")
    merge_gap_seconds = _finite(merge_gap_seconds, "merge_gap_seconds")
    min_duration_seconds = _finite(min_duration_seconds, "min_duration_seconds")
    vocal_ceiling = _finite(vocal_ceiling, "vocal_ceiling")
    hop_seconds = _finite(hop_seconds, "hop_seconds")
    if threshold < 0 or merge_gap_seconds < 0 or min_duration_seconds < 0 or vocal_ceiling < 0 or hop_seconds <= 0:
        raise ValueError("thresholds and durations must be non-negative; hop_seconds must be positive")
    events: list[dict[str, float]] = []
    start = last_active = None
    peak = 0.0
    last_time = -math.inf

    def finish() -> None:
        nonlocal start, last_active, peak
        if start is not None and last_active is not None and last_active - start + hop_seconds >= min_duration_seconds:
            events.append({
                "start_seconds": round(start, 3),
                "end_seconds": round(last_active + hop_seconds, 3),
                "peak_score": round(peak, 4),
            })
        start = last_active = None
        peak = 0.0

    for row in rows:
        time = _finite(row.get("time_seconds"), "time_seconds")
        if time <= last_time:
            raise ValueError("event rows must have strictly increasing time_seconds")
        last_time = time
        response = max(_finite(row.get(key, 0.0), key) for key in ("applause", "clapping", "cheering"))
        vocal = max(_finite(row.get(key, 0.0), key) for key in ("singing", "choir"))
        active = response >= threshold and vocal < vocal_ceiling
        if active:
            if start is not None and last_active is not None and time - last_active - hop_seconds > merge_gap_seconds:
                finish()
            if start is None:
                start = time
            last_active = time
            peak = max(peak, response)
        elif start is not None and last_active is not None and time - last_active - hop_seconds > merge_gap_seconds:
            finish()
    finish()
    return events


def validate_cuts(cuts: list[dict[str, Any]], *, source_duration: float) -> list[dict[str, Any]]:
    """Validate ordered, non-overlapping manual ranges against the source timeline."""
    duration = _finite(source_duration, "source duration")
    if duration <= 0 or not isinstance(cuts, list) or not cuts:
        raise ValueError("source duration and cuts must be positive/non-empty")
    validated = []
    previous_end = -1.0
    for index, cut in enumerate(cuts, 1):
        if not isinstance(cut, dict):
            raise ValueError(f"cut {index} must be an object")
        label = str(cut.get("label", "")).strip()
        start = _finite(cut.get("start_seconds"), f"cut {index} start_seconds")
        end = _finite(cut.get("end_seconds"), f"cut {index} end_seconds")
        if not label:
            raise ValueError(f"cut {index} needs a non-empty label")
        if start < 0 or end <= start:
            raise ValueError(f"cut {index} must have 0 <= start_seconds < end_seconds")
        if end > duration:
            raise ValueError(f"cut {index} exceeds source duration ({duration:.3f} seconds)")
        if start < previous_end:
            raise ValueError(f"cut {index} overlaps or is out of order")
        previous_end = end
        validated.append({**cut, "label": label, "start_seconds": start, "end_seconds": end})
    return validated


def _aligned_chunk_samples(chunk_seconds: int) -> int:
    if chunk_seconds < 2:
        raise ValueError("chunk_seconds must be at least 2")
    hop_samples = round(SAMPLE_RATE * HOP_SECONDS)
    chunk_hops = (chunk_seconds * SAMPLE_RATE + hop_samples - 1) // hop_samples
    return chunk_hops * hop_samples


def _stream_start_time(stream: dict[str, Any]) -> float:
    try:
        return float(stream.get("start_time") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _video_timeline(source_info: dict[str, Any]) -> tuple[float, float]:
    video = next((stream for stream in source_info["streams"] if stream.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("source needs a video stream")
    start = _stream_start_time(video)
    try:
        duration = float(video.get("duration"))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError
    except (TypeError, ValueError):
        format_start = _finite(source_info.get("format_start_time_seconds", 0.0), "format start time")
        duration = format_start + _finite(source_info["duration_seconds"], "source duration") - start
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("video stream duration must be positive and finite")
    return start, duration


def _ffmpeg_seek_seconds(source_info: dict[str, Any], video_relative_seconds: float) -> float:
    video_start, _ = _video_timeline(source_info)
    format_start = _finite(source_info.get("format_start_time_seconds", 0.0), "format start time")
    relative = _finite(video_relative_seconds, "video-relative seek time")
    return video_start - format_start + relative


def _audio_start_offset(streams: list[dict[str, Any]]) -> float:
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise ValueError("source needs both audio and video streams")
    return _stream_start_time(audio) - _stream_start_time(video)


def _read_exact(stream: Any, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        part = stream.read(length - len(chunks))
        if not part:
            break
        chunks.extend(part)
    return bytes(chunks)


def scan_video_events(
    video_path: Path, model_path: Path, output_csv: Path, *, chunk_seconds: int = 30,
    threshold: float = 0.18, merge_gap_seconds: float = 3.0,
    min_duration_seconds: float = 1.5, vocal_ceiling: float = 0.25,
) -> dict[str, Any]:
    """Run YAMNet ONNX over the full soundtrack and write scores plus reproducibility metadata."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install song segmentation dependencies with: pip install -e '.[song-segmentation]'") from exc
    source = video_path.resolve(strict=True)
    model = model_path.resolve(strict=True)
    if not source.is_file() or not model.is_file():
        raise ValueError("video_path and model_path must refer to files")
    source_info = probe_media(source)
    video_start, video_duration = _video_timeline(source_info)
    audio_offset = _audio_start_offset(source_info["streams"])
    metadata_path = output_csv.with_suffix(".metadata.json")
    if output_csv.exists() or metadata_path.exists():
        raise FileExistsError(f"Output CSV or metadata already exists; refusing to overwrite: {output_csv}")
    if chunk_seconds < 2:
        raise ValueError("chunk_seconds must be at least 2")
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    command = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(source),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    stderr_file = tempfile.TemporaryFile()
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file)
    except OSError:
        stderr_file.close()
        raise
    if process.stdout is None:
        process.kill()
        process.wait()
        stderr_file.close()
        raise RuntimeError("could not open FFmpeg audio stream")
    fields = ["time_seconds", *CLASS_IDS.keys()]
    rows_written = 0
    try:
        with output_csv.open("x", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            hop_samples = round(SAMPLE_RATE * HOP_SECONDS)
            chunk_samples = _aligned_chunk_samples(chunk_seconds)
            overlap_samples = 2 * hop_samples
            carry = np.empty(0, dtype=np.float32)
            consumed_samples = 0
            last_frame_time = -math.inf
            while True:
                raw = _read_exact(process.stdout, chunk_samples * 4)
                usable_bytes = len(raw) - len(raw) % 4
                if usable_bytes == 0:
                    break
                waveform = np.frombuffer(raw[:usable_bytes], dtype=np.float32)
                inference_waveform = np.concatenate((carry, waveform)) if carry.size else waveform
                inference_start_sample = consumed_samples - len(carry)
                scores = np.asarray(session.run([output_name], {input_name: inference_waveform})[0])
                for frame, score in enumerate(scores):
                    frame_time = audio_offset + (inference_start_sample + frame * hop_samples) / SAMPLE_RATE
                    if frame_time <= last_frame_time + 1e-6:
                        continue
                    row = {"time_seconds": f"{frame_time:.3f}"}
                    row.update({name: f"{float(score[class_id]):.6f}" for name, class_id in CLASS_IDS.items()})
                    writer.writerow(row)
                    rows_written += 1
                    last_frame_time = frame_time
                carry = waveform[-overlap_samples:]
                consumed_samples += len(waveform)
        return_code = process.wait()
        process.stdout.close()
        if return_code != 0:
            stderr_file.seek(0)
            stderr = stderr_file.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg audio decode failed: {stderr[-2000:]}")
        with output_csv.open(newline="", encoding="utf-8") as source_csv:
            candidates = applause_spans(
                csv.DictReader(source_csv), threshold=threshold, merge_gap_seconds=merge_gap_seconds,
                min_duration_seconds=min_duration_seconds, vocal_ceiling=vocal_ceiling,
            )
        ffmpeg_version = run_cmd(["ffmpeg", "-version"], check=False).stdout.splitlines()[0]
        metadata = {
            "schema_version": 1, "source_file": str(source), "source_sha256": _sha256(source),
            "source_duration_seconds": video_duration, "video_stream_start_time_seconds": video_start,
            "format_start_time_seconds": source_info["format_start_time_seconds"],
            "timeline_origin": "first_video_stream_start", "audio_start_offset_seconds": audio_offset,
            "model_file": str(model), "model_sha256": _sha256(model), "ffmpeg_version": ffmpeg_version,
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "onnxruntime_version": ort.__version__,
            "sample_rate": SAMPLE_RATE, "chunk_seconds_requested": chunk_seconds,
            "chunk_seconds_actual": chunk_samples / SAMPLE_RATE,
            "context_overlap_seconds": overlap_samples / SAMPLE_RATE,
            "frame_hop_seconds": HOP_SECONDS,
            "class_ids": CLASS_IDS, "frame_count": rows_written,
            "candidate_settings": {"threshold": threshold, "merge_gap_seconds": merge_gap_seconds,
                                   "min_duration_seconds": min_duration_seconds, "vocal_ceiling": vocal_ceiling},
            "candidate_applause_spans": candidates,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return metadata
    except Exception:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        output_csv.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise
    finally:
        stderr_file.close()


def _video_bitrate(duration: float, max_size_mib: float, audio_bitrate: int) -> int:
    if not math.isfinite(max_size_mib) or max_size_mib <= 0:
        raise ValueError("max_size_mib must be positive")
    # Reserve 15% for muxing variation and VBR peaks; final file size is checked after encoding.
    budget = int(max_size_mib * 1024 * 1024 * 0.85)
    bitrate = int(budget * 8 / duration - audio_bitrate)
    if bitrate < 200_000:
        raise ValueError("target size is too small for this clip duration")
    return bitrate


def render_song_cuts(
    video_path: Path, cuts: list[dict[str, Any]], output_dir: Path, *, height: int = 720,
    max_size_mib: float = 95.0, fade_seconds: float = 3.0, encoder: str = "h264_nvenc",
    preset: Optional[str] = None, cardwire_gpu: Optional[str] = None, audio_bitrate: int = 128_000,
) -> dict[str, Any]:
    """Render reviewed ranges with an audio tail fade; refuse overwrites and verify output sizes."""
    source = video_path.resolve(strict=True)
    source_info = probe_media(source)
    streams = source_info["streams"]
    if not any(s.get("codec_type") == "audio" for s in streams):
        raise ValueError("source must contain both video and audio streams")
    video_start, video_duration = _video_timeline(source_info)
    if height <= 0 or height % 2:
        raise ValueError("height must be a positive even number")
    fade_seconds = _finite(fade_seconds, "fade_seconds")
    max_size_mib = _finite(max_size_mib, "max_size_mib")
    if fade_seconds < 0 or audio_bitrate <= 0:
        raise ValueError("fade_seconds must be non-negative and audio_bitrate positive")
    if max_size_mib <= 0:
        raise ValueError("max_size_mib must be positive")
    preset = preset or ("p5" if encoder == "h264_nvenc" else "medium")
    if not preset.strip():
        raise ValueError("preset must not be empty")
    validated = validate_cuts(cuts, source_duration=video_duration)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir.parent / f".{output_dir.name}.partial"
    if temporary_dir.exists():
        raise FileExistsError(f"Temporary output already exists; remove it before retrying: {temporary_dir}")
    temporary_dir.mkdir()
    manifest_cuts = []
    try:
        for index, cut in enumerate(validated, 1):
            duration = cut["end_seconds"] - cut["start_seconds"]
            bitrate = _video_bitrate(duration, max_size_mib, audio_bitrate)
            fade = min(fade_seconds, duration)
            output_name = f"song-{index:02d}.mp4"
            partial = temporary_dir / output_name
            ffmpeg_seek_seconds = _ffmpeg_seek_seconds(source_info, cut["start_seconds"])
            command = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{ffmpeg_seek_seconds:.3f}",
                       "-i", str(source), "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "0:a:0",
                       "-vf", f"scale=-2:{height}:flags=lanczos,format=yuv420p", "-c:v", encoder, "-preset", preset,
                       "-b:v", str(bitrate), "-maxrate", str(bitrate), "-bufsize", str(bitrate * 2),
                       "-c:a", "aac", "-b:a", str(audio_bitrate)]
            if fade > 0:
                command.extend(["-af", f"afade=t=out:st={duration - fade:.3f}:d={fade:.3f}"])
            command.extend(["-movflags", "+faststart", str(partial)])
            if cardwire_gpu is not None:
                command = ["cardwire", "launch", "--gpu", str(cardwire_gpu), *command]
            result = run_cmd(command, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"Rendering {output_name} failed:\n{result.stdout[-2000:]}")
            if not partial.is_file():
                raise RuntimeError(f"FFmpeg reported success but did not create {output_name}")
            size_bytes = partial.stat().st_size
            if size_bytes > max_size_mib * 1024 * 1024:
                raise RuntimeError(f"{output_name} is {size_bytes} bytes, above the {max_size_mib:g} MiB target; lower the target size or split the cut")
            rendered = probe_media(partial)
            rendered_video = next((s for s in rendered["streams"] if s.get("codec_type") == "video"), None)
            rendered_audio = any(s.get("codec_type") == "audio" for s in rendered["streams"])
            if rendered_video is None or rendered_video.get("height") != height or not rendered_audio:
                raise RuntimeError(f"Rendered clip {output_name} does not have the requested video height and audio stream")
            if abs(rendered["duration_seconds"] - duration) > 0.1:
                raise RuntimeError(f"Rendered clip {output_name} duration differs from requested range by more than 0.1 seconds")
            decoded = run_cmd(["ffmpeg", "-v", "error", "-i", str(partial), "-f", "null", "-"], check=False)
            if decoded.returncode != 0:
                raise RuntimeError(f"Rendered clip {output_name} failed full decode:\n{decoded.stdout[-2000:]}")
            manifest_cuts.append({
                "label": cut["label"], "file": output_name,
                "start_seconds": round(cut["start_seconds"], 3), "end_seconds": round(cut["end_seconds"], 3),
                "duration_seconds": round(duration, 3), "size_bytes": size_bytes,
                "width": rendered_video.get("width"), "height": rendered_video.get("height"),
                "encoder": encoder, "preset": preset, "video_bitrate": bitrate,
                "audio_fade_seconds": round(fade, 3),
                "boundary_evidence": cut.get("boundary_evidence", "manual cut list"),
            })
        manifest = {
            "schema_version": 1, "source_file": str(source), "source_sha256": _sha256(source),
            "source_duration_seconds": video_duration,
            "video_stream_start_time_seconds": video_start,
            "format_start_time_seconds": source_info["format_start_time_seconds"],
            "ffmpeg_version": run_cmd(["ffmpeg", "-version"], check=False).stdout.splitlines()[0],
            "python_version": platform.python_version(),
            "detector_note": "Automated event scores suggest boundaries; reviewed cut list controls all rendering.",
            "max_size_mib": max_size_mib, "cardwire_gpu": cardwire_gpu, "clips": manifest_cuts,
        }
        (temporary_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary_dir.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
