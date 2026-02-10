from __future__ import annotations

from .config import ASRConfig, Features, PipelineConfig


def aggressive() -> PipelineConfig:
    return PipelineConfig(
        features=Features(
            enhance=True,
            make_asr_wav=False,
            transcribe=True,
            diarize=False,
            subtitles=True,
            master_outputs=True,
        ),
        asr=ASRConfig(
            model="medium",
            device="cpu",
            compute_type="int8",
            beam_size=7,
            language=None,
            vad=False,
            vad_min_silence_ms=500,
            temperature=0.0,
        ),
        mono=True,
        force=False,
    )


def balanced() -> PipelineConfig:
    cfg = aggressive()
    cfg.features.make_asr_wav = True
    cfg.asr.beam_size = 5
    cfg.asr.vad = True
    cfg.asr.vad_min_silence_ms = 700
    return cfg


def gentle() -> PipelineConfig:
    cfg = aggressive()
    cfg.features.enhance = False
    cfg.features.make_asr_wav = False
    cfg.asr.beam_size = 3
    cfg.asr.vad = False
    cfg.asr.vad_min_silence_ms = 1000
    return cfg


PROFILES = {
    "aggressive": aggressive,
    "balanced": balanced,
    "gentle": gentle,
}
