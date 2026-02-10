from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Features:
    enhance: bool = True
    make_asr_wav: bool = False
    transcribe: bool = True
    diarize: bool = False
    subtitles: bool = True
    master_outputs: bool = True


@dataclass
class ASRConfig:
    model: str = "medium"
    device: str = "cpu"
    compute_type: Optional[str] = "int8"
    beam_size: int = 7
    language: Optional[str] = None
    vad: bool = False
    vad_min_silence_ms: int = 500
    temperature: float = 0.0


@dataclass
class DiarizationConfig:
    max_speakers: int = 6
    vad_aggr: int = 2


@dataclass
class PipelineConfig:
    features: Features = field(default_factory=Features)
    asr: ASRConfig = field(default_factory=ASRConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    mono: bool = True
    force: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_config(config: PipelineConfig) -> None:
    if config.asr.beam_size <= 0:
        raise ValueError("Invalid config: asr.beam_size must be > 0")
    if config.asr.vad_min_silence_ms < 0:
        raise ValueError("Invalid config: asr.vad_min_silence_ms must be >= 0")
    if config.diarization.max_speakers <= 0:
        raise ValueError("Invalid config: diarization.max_speakers must be > 0")
    if config.diarization.vad_aggr not in (0, 1, 2, 3):
        raise ValueError("Invalid config: diarization.vad_aggr must be one of 0,1,2,3")
    if config.features.diarize and not config.features.transcribe:
        raise ValueError("Invalid config: diarization requires transcription to be enabled")
