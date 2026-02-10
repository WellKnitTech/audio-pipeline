from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable, Optional

from .enhance import enhance_to_m4a, make_asr_wav_16k
from .transcribe import transcribe_file
from .utils import ensure_ffmpeg, safe_mkdir

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus", ".mp4", ".mov"}

def iter_audio_files(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p

def run_pipeline(
    input_dir: Path,
    output_root: Path,
    *,
    force: bool,
    mono: bool,
    make_asr_wav: bool,
    model: str,
    device: str,
    compute_type: Optional[str],
    beam_size: int,
    language: Optional[str],
    vad: bool,
    vad_min_silence_ms: int,
    temperature: float,
    diarize: bool,
    max_speakers: int,
    vad_aggr: int,
) -> None:
    ensure_ffmpeg()

    safe_mkdir(output_root)
    enhanced_dir = output_root / "enhanced"
    asr_wav_dir = output_root / "asr_wav"
    transcripts_dir = output_root / "transcripts"

    safe_mkdir(enhanced_dir)
    safe_mkdir(transcripts_dir)
    if make_asr_wav or diarize:
        safe_mkdir(asr_wav_dir)

    master_txt_path = output_root / "master_transcript.txt"
    master_csv_path = output_root / "master_transcripts.csv"

    master_rows = []
    master_text_blocks = []

    files = list(iter_audio_files(input_dir))
    if not files:
        logging.warning("No supported audio files found in %s", input_dir)
        return

    diarizer = None
    if diarize:
        from . import diarize as diarize_mod
        diarizer = diarize_mod

    for in_file in files:
        stem = in_file.stem
        enhanced_path = enhanced_dir / f"{stem}.enhanced.m4a"

        enhance_to_m4a(in_file, enhanced_path, force=force, mono=mono)

        asr_wav_path = None
        if make_asr_wav or diarize:
            asr_wav_path = asr_wav_dir / f"{enhanced_path.stem}.asr16k.wav"
            make_asr_wav_16k(enhanced_path, asr_wav_path, force=force)

        transcribe_input = asr_wav_path if asr_wav_path else enhanced_path

        txt_path, srt_path, vtt_path, json_path, seg_count = transcribe_file(
            transcribe_input,
            transcripts_dir,
            model_name=model,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            language=language,
            vad=vad,
            vad_min_silence_ms=vad_min_silence_ms,
            temperature=temperature,
            force=force,
        )

        speaker_txt = ""
        speaker_srt = ""
        rttm = ""

        if diarize and diarizer is not None:
            if asr_wav_path is None:
                logging.error("Diarization requested but ASR wav was not created.")
            else:
                try:
                    rttm_path, speaker_txt_path, speaker_srt_path = diarizer.diarize_and_merge(
                        asr_wav16k=asr_wav_path,
                        whisper_json=json_path,
                        out_dir=transcripts_dir,
                        max_speakers=max_speakers,
                        vad_aggr=vad_aggr,
                        force=force,
                    )
                    rttm = str(rttm_path)
                    speaker_txt = str(speaker_txt_path)
                    speaker_srt = str(speaker_srt_path)
                except Exception as e:
                    logging.error("Diarization failed for %s: %s", in_file.name, e)

        text = txt_path.read_text(encoding="utf-8").strip()
        master_text_blocks.append(f"===== {in_file.name} =====\n{text}\n")
        master_rows.append({
            "source_file": in_file.name,
            "enhanced_file": str(enhanced_path),
            "asr_wav": str(asr_wav_path) if asr_wav_path else "",
            "txt": str(txt_path),
            "srt": str(srt_path),
            "vtt": str(vtt_path),
            "json": str(json_path),
            "segments": seg_count,
            "speaker_txt": speaker_txt,
            "speaker_srt": speaker_srt,
            "rttm": rttm,
        })

    master_txt_path.write_text("\n".join(master_text_blocks).rstrip() + "\n", encoding="utf-8")

    fieldnames = [
        "source_file", "enhanced_file", "asr_wav", "txt", "srt", "vtt", "json", "segments",
        "speaker_txt", "speaker_srt", "rttm"
    ]
    with master_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in master_rows:
            w.writerow(row)
