from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from .utils import run_cmd


def probe_media(media_path: Path) -> Dict[str, Any]:
    if not media_path.is_file():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels,start_time",
        "-of", "json", str(media_path),
    ]
    result = run_cmd(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {media_path}:\n{result.stdout}")
    try:
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ffprobe returned invalid media metadata for {media_path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Media duration must be positive: {media_path}")

    return {
        "duration_seconds": duration,
        "size_bytes": int(data.get("format", {}).get("size") or media_path.stat().st_size),
        "streams": data.get("streams", []),
    }


def _stream_start_time(stream: Dict[str, Any]) -> float:
    try:
        return float(stream.get("start_time") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def prepare_video(video_path: Path, output_dir: Path, chunk_seconds: int = 600) -> Dict[str, Any]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be greater than zero")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite: {output_dir}")

    source = video_path.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Video source is not a file: {source}")
    source_metadata = probe_media(source)
    audio_streams = [stream for stream in source_metadata["streams"] if stream.get("codec_type") == "audio"]
    video_streams = [stream for stream in source_metadata["streams"] if stream.get("codec_type") == "video"]
    if not audio_streams:
        raise ValueError(f"Video has no audio stream: {source}")
    if not video_streams:
        raise ValueError(f"Input has no video stream: {source}")
    audio_start_offset = _stream_start_time(audio_streams[0]) - _stream_start_time(video_streams[0])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    output_created = False
    committed = False
    try:
        audio_dir = temporary_dir / "audio"
        audio_dir.mkdir()
        pattern = audio_dir / "chunk_%04d.wav"
        command = [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-i", str(source), "-map", "0:a:0", "-vn",
            "-f", "segment", "-segment_time", str(chunk_seconds),
            "-reset_timestamps", "1", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(pattern),
        ]
        result = run_cmd(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed for {source}:\n{result.stdout}")

        chunk_files = sorted(audio_dir.glob("chunk_*.wav"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
        if not chunk_files:
            raise RuntimeError(f"ffmpeg produced no audio chunks for {source}")

        chunks: List[Dict[str, Any]] = []
        offset = audio_start_offset
        for chunk_path in chunk_files:
            chunk_metadata = probe_media(chunk_path)
            duration = chunk_metadata["duration_seconds"]
            chunks.append({
                "file": str(Path("audio") / chunk_path.name),
                "start_seconds": round(offset, 6),
                "duration_seconds": round(duration, 6),
                "end_seconds": round(offset + duration, 6),
                "size_bytes": chunk_metadata["size_bytes"],
            })
            offset += duration

        manifest = {
            "schema_version": 1,
            "source_file": str(source),
            "source_size_bytes": source_metadata["size_bytes"],
            "duration_seconds": source_metadata["duration_seconds"],
            "timeline_origin": "first_video_stream_start",
            "audio_start_offset_seconds": round(audio_start_offset, 6),
            "streams": source_metadata["streams"],
            "audio_format": {"codec": "pcm_s16le", "sample_rate": 16000, "channels": 1},
            "chunk_seconds_requested": chunk_seconds,
            "chunks": chunks,
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        output_dir.mkdir()
        output_created = True
        os.replace(audio_dir, output_dir / "audio")
        os.replace(temporary_dir / "manifest.json", output_dir / "manifest.json")
        committed = True
        return manifest
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        if output_created and not committed:
            shutil.rmtree(output_dir, ignore_errors=True)


def merge_chunk_transcripts(manifest_path: Path, transcript_dir: Path, output_path: Path) -> Dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"Output file already exists; refusing to overwrite: {output_path}")
    if not transcript_dir.is_dir():
        raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = manifest.get("chunks") if isinstance(manifest, dict) else None
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Manifest must contain a non-empty chunks list")

    segments: List[Dict[str, Any]] = []
    sources = []
    for chunk in chunks:
        if not isinstance(chunk, dict) or not chunk.get("file"):
            raise ValueError("Each manifest chunk must include a file path")
        chunk_name = Path(str(chunk["file"])).name
        chunk_stem = Path(chunk_name).stem
        try:
            offset = float(chunk["start_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Chunk {chunk_name} is missing a valid start_seconds offset") from exc
        if not math.isfinite(offset):
            raise ValueError(f"Chunk {chunk_name} has a non-finite start_seconds offset")

        candidates = [
            transcript_dir / f"{chunk_stem}.json",
            transcript_dir / f"{chunk_stem}.asr16k.json",
            transcript_dir / f"{chunk_stem}.enhanced.json",
            transcript_dir / f"{chunk_stem}.enhanced.asr16k.json",
        ]
        available = [path for path in candidates if path.is_file()]
        if not available:
            raise FileNotFoundError(f"No transcript JSON found for chunk {chunk_name} in {transcript_dir}")
        if len(available) > 1:
            raise ValueError(f"Multiple transcript JSON files found for chunk {chunk_name}: {available}")
        transcript_path = available[0]
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        local_segments = transcript.get("segments") if isinstance(transcript, dict) else None
        if not isinstance(local_segments, list):
            raise ValueError(f"Transcript must contain a segments list: {transcript_path}")

        for index, segment in enumerate(local_segments):
            try:
                start = float(segment["start"])
                end = float(segment["end"])
                text = str(segment["text"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid segment {index} in {transcript_path}") from exc
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                raise ValueError(f"Invalid segment times at index {index} in {transcript_path}")
            global_start = start + offset
            global_end = end + offset
            if not math.isfinite(global_start) or not math.isfinite(global_end):
                raise ValueError(f"Rebased segment times overflow at index {index} in {transcript_path}")
            if text:
                segments.append({
                    "start": round(global_start, 6),
                    "end": round(global_end, 6),
                    "text": text,
                    "chunk": chunk_name,
                })
        sources.append({"chunk": chunk_name, "transcript": transcript_path.name, "offset_seconds": offset})

    segments.sort(key=lambda segment: (segment["start"], segment["end"], segment["chunk"]))
    result = {
        "meta": {
            "source_file": manifest.get("source_file"),
            "duration_seconds": manifest.get("duration_seconds"),
            "timeline_origin": manifest.get("timeline_origin", "first_video_stream_start"),
            "chunk_count": len(chunks),
            "transcript_sources": sources,
        },
        "segments": segments,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return result
