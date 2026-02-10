from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from faster_whisper import WhisperModel


@dataclass
class SegmentOut:
    start: float
    end: float
    text: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    hh = ms // 3_600_000
    ms -= hh * 3_600_000
    mm = ms // 60_000
    ms -= mm * 60_000
    ss = ms // 1_000
    ms -= ss * 1_000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def vtt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    hh = ms // 3_600_000
    ms -= hh * 3_600_000
    mm = ms // 60_000
    ms -= mm * 60_000
    ss = ms // 1_000
    ms -= ss * 1_000
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def write_txt(path: Path, segments: List[SegmentOut]) -> None:
    text = " ".join(s.text.strip() for s in segments).strip()
    path.write_text(text + "\n", encoding="utf-8")


def write_srt(path: Path, segments: List[SegmentOut]) -> None:
    lines: List[str] = []
    for i, s in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{srt_timestamp(s.start)} --> {srt_timestamp(s.end)}")
        lines.append(s.text.strip())
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_vtt(path: Path, segments: List[SegmentOut]) -> None:
    lines: List[str] = ["WEBVTT", ""]
    for s in segments:
        lines.append(f"{vtt_timestamp(s.start)} --> {vtt_timestamp(s.end)}")
        lines.append(s.text.strip())
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, meta: dict, segments: List[SegmentOut]) -> None:
    payload = {"meta": meta, "segments": [asdict(s) for s in segments]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def transcribe_file(
    audio_path: Path,
    out_dir: Path,
    model_name: str,
    device: str,
    compute_type: Optional[str],
    beam_size: int,
    language: Optional[str],
    vad: bool,
    vad_min_silence_ms: int,
    temperature: float,
    force: bool,
    subtitles: bool,
) -> Tuple[Path, Optional[Path], Optional[Path], Path, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    txt_path = out_dir / f"{stem}.txt"
    srt_path = out_dir / f"{stem}.srt"
    vtt_path = out_dir / f"{stem}.vtt"
    json_path = out_dir / f"{stem}.json"

    required = [txt_path, json_path] + ([srt_path, vtt_path] if subtitles else [])
    if not force and all(p.exists() and p.stat().st_size > 0 for p in required):
        logging.info("SKIP transcript exists: %s", audio_path.name)
        return txt_path, (srt_path if subtitles else None), (vtt_path if subtitles else None), json_path, 0

    logging.info("Transcribing: %s", audio_path.name)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    vad_params = {"min_silence_duration_ms": vad_min_silence_ms} if vad else None

    segments_iter, info = model.transcribe(
        str(audio_path),
        log_progress=True,
        beam_size=beam_size,
        language=language,
        vad_filter=vad,
        vad_parameters=vad_params,
        temperature=temperature,
        condition_on_previous_text=True,
    )

    segments: List[SegmentOut] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if text:
            segments.append(SegmentOut(start=float(seg.start), end=float(seg.end), text=text))

    write_txt(txt_path, segments)
    if subtitles:
        write_srt(srt_path, segments)
        write_vtt(vtt_path, segments)

    meta = {
        "source_file": audio_path.name,
        "source_path": str(audio_path.resolve()),
        "transcribed_utc": utc_now_iso() + "Z",
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": language or getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "beam_size": beam_size,
        "vad_filter": vad,
        "vad_min_silence_ms": vad_min_silence_ms if vad else None,
        "temperature": temperature,
        "segment_count": len(segments),
    }
    write_json(json_path, meta, segments)

    return txt_path, (srt_path if subtitles else None), (vtt_path if subtitles else None), json_path, len(segments)
