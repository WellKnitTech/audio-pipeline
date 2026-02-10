from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from .utils import ffmpeg_filters, has_filter, run_cmd, safe_mkdir

# Aggressive loudness defaults (back-of-room iPhone style audio):
# - Compressor grabs quiet speech and lifts it
# - loudnorm targets a louder program level
DEFAULT_COMP = "acompressor=threshold=-20dB:ratio=4:attack=5:release=250:makeup=12"
DEFAULT_LOUDNORM = "loudnorm=I=-12:TP=-1.0:LRA=11"
DEFAULT_HIGHPASS = "highpass=f=80"

def build_audio_filter_chain() -> Tuple[str, str]:
    filters = ffmpeg_filters()
    has_loudnorm = has_filter(filters, "loudnorm")
    has_dynaudnorm = has_filter(filters, "dynaudnorm")
    has_alimiter = has_filter(filters, "alimiter")

    if has_loudnorm:
        af = f"{DEFAULT_HIGHPASS},{DEFAULT_COMP},{DEFAULT_LOUDNORM}"
        return af, "loudnorm"
    if has_dynaudnorm and has_alimiter:
        # Fallback: still loud, but not LUFS-accurate
        af = f"{DEFAULT_HIGHPASS},{DEFAULT_COMP},dynaudnorm=f=150:g=20:p=0.95:m=10,alimiter=limit=-1.0dB"
        return af, "dynaudnorm+alimiter"
    if has_dynaudnorm:
        af = f"{DEFAULT_HIGHPASS},{DEFAULT_COMP},dynaudnorm=f=150:g=20:p=0.95:m=10"
        return af, "dynaudnorm"
    raise RuntimeError("ffmpeg is missing loudnorm and dynaudnorm filters; cannot enhance audio.")

def enhance_to_m4a(
    in_file: Path,
    out_file: Path,
    force: bool = False,
    mono: bool = True,
    bitrate: str = "192k",
) -> bool:
    if out_file.exists() and out_file.stat().st_size > 0 and not force:
        logging.info("SKIP enhanced exists: %s", out_file.name)
        return False

    safe_mkdir(out_file.parent)
    afilter, mode = build_audio_filter_chain()
    logging.info("Enhance mode=%s: %s", mode, in_file.name)

    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-i", str(in_file),
        "-vn",
        "-af", afilter,
    ]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-c:a", "aac", "-b:a", bitrate, "-movflags", "+faststart", str(out_file)]

    cp = run_cmd(cmd, check=False)
    if cp.returncode != 0:
        logging.error("Enhance failed for %s:\n%s", in_file.name, cp.stdout)
        return False
    return True

def make_asr_wav_16k(
    in_file: Path,
    out_wav: Path,
    force: bool = False,
) -> bool:
    if out_wav.exists() and out_wav.stat().st_size > 0 and not force:
        logging.info("SKIP asr wav exists: %s", out_wav.name)
        return False

    safe_mkdir(out_wav.parent)
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-i", str(in_file),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    cp = run_cmd(cmd, check=False)
    if cp.returncode != 0:
        logging.error("ASR wav conversion failed for %s:\n%s", in_file.name, cp.stdout)
        return False
    return True
