from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

from .config import PipelineConfig
from .enhance import enhance_to_m4a, make_asr_wav_16k
from .transcribe import transcribe_file
from .utils import ensure_ffmpeg, safe_mkdir

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus", ".mp4", ".mov"}


def iter_audio_files(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def enabled_stage_names(config: PipelineConfig) -> list[str]:
    stages: list[str] = []
    if config.features.enhance:
        stages.append("enhance")
    if config.features.make_asr_wav or config.features.diarize:
        stages.append("make_asr_wav")
    if config.features.transcribe:
        stages.append("transcribe")
    if config.features.diarize:
        stages.append("diarize")
    if config.features.subtitles:
        stages.append("subtitles")
    if config.features.master_outputs:
        stages.append("master_outputs")
    return stages


def run_pipeline(input_dir: Path, output_root: Path, config: PipelineConfig) -> None:
    ensure_ffmpeg()

    safe_mkdir(output_root)
    enhanced_dir = output_root / "enhanced"
    asr_wav_dir = output_root / "asr_wav"
    transcripts_dir = output_root / "transcripts"

    if config.features.enhance:
        safe_mkdir(enhanced_dir)
    if config.features.transcribe or config.features.diarize:
        safe_mkdir(transcripts_dir)
    if config.features.make_asr_wav or config.features.diarize:
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
    if config.features.diarize:
        from . import diarize as diarize_mod

        diarizer = diarize_mod

    for in_file in files:
        stem = in_file.stem
        stage_input = in_file
        enhanced_path = None

        if config.features.enhance:
            enhanced_path = enhanced_dir / f"{stem}.enhanced.m4a"
            enhance_to_m4a(in_file, enhanced_path, force=config.force, mono=config.mono)
            stage_input = enhanced_path

        asr_wav_path = None
        if config.features.make_asr_wav or config.features.diarize:
            wav_stem = enhanced_path.stem if enhanced_path else in_file.stem
            asr_wav_path = asr_wav_dir / f"{wav_stem}.asr16k.wav"
            make_asr_wav_16k(stage_input, asr_wav_path, force=config.force)

        transcribe_input = asr_wav_path if asr_wav_path else stage_input

        txt_path = None
        srt_path = None
        vtt_path = None
        json_path = None
        seg_count = 0

        if config.features.transcribe:
            txt_path, srt_path, vtt_path, json_path, seg_count = transcribe_file(
                transcribe_input,
                transcripts_dir,
                model_name=config.asr.model,
                device=config.asr.device,
                compute_type=config.asr.compute_type,
                beam_size=config.asr.beam_size,
                language=config.asr.language,
                vad=config.asr.vad,
                vad_min_silence_ms=config.asr.vad_min_silence_ms,
                temperature=config.asr.temperature,
                force=config.force,
                subtitles=config.features.subtitles,
            )

        speaker_txt = ""
        speaker_srt = ""
        rttm = ""

        if config.features.diarize and diarizer is not None:
            if asr_wav_path is None or json_path is None:
                logging.error("Diarization requested but required transcription inputs are unavailable.")
            else:
                try:
                    rttm_path, speaker_txt_path, speaker_srt_path = diarizer.diarize_and_merge(
                        asr_wav16k=asr_wav_path,
                        whisper_json=json_path,
                        out_dir=transcripts_dir,
                        max_speakers=config.diarization.max_speakers,
                        vad_aggr=config.diarization.vad_aggr,
                        force=config.force,
                    )
                    rttm = str(rttm_path)
                    speaker_txt = str(speaker_txt_path)
                    speaker_srt = str(speaker_srt_path)
                except Exception as e:
                    logging.error("Diarization failed for %s: %s", in_file.name, e)

        if config.features.master_outputs and txt_path is not None:
            text = txt_path.read_text(encoding="utf-8").strip()
            master_text_blocks.append(f"===== {in_file.name} =====\n{text}\n")
            master_rows.append(
                {
                    "source_file": in_file.name,
                    "enhanced_file": str(enhanced_path) if enhanced_path else "",
                    "asr_wav": str(asr_wav_path) if asr_wav_path else "",
                    "txt": str(txt_path),
                    "srt": str(srt_path) if srt_path else "",
                    "vtt": str(vtt_path) if vtt_path else "",
                    "json": str(json_path) if json_path else "",
                    "segments": seg_count,
                    "speaker_txt": speaker_txt,
                    "speaker_srt": speaker_srt,
                    "rttm": rttm,
                }
            )

    if config.features.master_outputs and master_text_blocks:
        master_txt_path.write_text("\n".join(master_text_blocks).rstrip() + "\n", encoding="utf-8")

        fieldnames = [
            "source_file",
            "enhanced_file",
            "asr_wav",
            "txt",
            "srt",
            "vtt",
            "json",
            "segments",
            "speaker_txt",
            "speaker_srt",
            "rttm",
        ]
        with master_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in master_rows:
                w.writerow(row)
