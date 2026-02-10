from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

try:
    import numpy as np
    import soundfile as sf
    import webrtcvad
    from resemblyzer import VoiceEncoder
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
except Exception:
    np = None  # type: ignore

@dataclass
class WhisperSeg:
    start: float
    end: float
    text: str

@dataclass
class DiarSeg:
    start: float
    end: float
    spk: str

def deps_available() -> bool:
    return np is not None

def read_whisper_json(path: Path) -> List[WhisperSeg]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = []
    for s in data.get("segments", []):
        segs.append(WhisperSeg(float(s["start"]), float(s["end"]), (s.get("text") or "").strip()))
    return [s for s in segs if s.text]

def frame_generator(pcm16: "np.ndarray", sr: int, frame_ms: int) -> List["np.ndarray"]:
    frame_len = int(sr * frame_ms / 1000)
    frames = []
    for i in range(0, len(pcm16) - frame_len + 1, frame_len):
        frames.append(pcm16[i:i + frame_len])
    return frames

def vad_segments(wav_path: Path, aggressiveness: int = 2, frame_ms: int = 30,
                 min_speech_ms: int = 400, min_silence_ms: int = 500) -> Tuple["np.ndarray", List[Tuple[float, float]]]:
    audio, sr = sf.read(str(wav_path), dtype="int16")
    if sr != 16000:
        raise ValueError("Expected 16kHz wav for diarization.")
    if getattr(audio, "ndim", 1) != 1:
        audio = audio[:, 0]

    vad = webrtcvad.Vad(aggressiveness)
    frames = frame_generator(audio, sr, frame_ms)
    is_speech = [vad.is_speech(f.tobytes(), sr) for f in frames]

    frame_dur = frame_ms / 1000.0
    segments = []
    in_seg = False
    seg_start = 0.0
    silence = 0.0

    for idx, sp in enumerate(is_speech):
        t = idx * frame_dur
        if sp:
            if not in_seg:
                in_seg = True
                seg_start = t
                silence = 0.0
            else:
                silence = 0.0
        else:
            if in_seg:
                silence += frame_dur
                if silence >= (min_silence_ms / 1000.0):
                    segments.append((seg_start, t))
                    in_seg = False
                    silence = 0.0

    if in_seg:
        segments.append((seg_start, len(is_speech) * frame_dur))

    merged = []
    for s, e in segments:
        if (e - s) < (min_speech_ms / 1000.0):
            continue
        if not merged:
            merged.append([s, e]); continue
        if s - merged[-1][1] <= 0.2:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    audio_f32 = audio.astype("float32") / 32768.0
    return audio_f32, [(float(a), float(b)) for a, b in merged]

def slice_audio(audio_f32: "np.ndarray", sr: int, start: float, end: float) -> "np.ndarray":
    a = max(0, int(start * sr))
    b = min(len(audio_f32), int(end * sr))
    return audio_f32[a:b]

def choose_k(embs: "np.ndarray", k_min: int, k_max: int) -> int:
    if len(embs) < 3:
        return 1
    best_k = 2
    best_score = -1.0
    for k in range(k_min, min(k_max, len(embs)) + 1):
        try:
            labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(embs)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embs, labels, metric="cosine")
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue
    return best_k

def diarize_and_merge(
    asr_wav16k: Path,
    whisper_json: Path,
    out_dir: Path,
    max_speakers: int = 6,
    vad_aggr: int = 2,
    force: bool = False,
) -> Tuple[Path, Path, Path]:
    if not deps_available():
        raise RuntimeError("Diarization deps not installed. Run: pip install -e \".[diarization]\"")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = asr_wav16k.stem

    rttm_path = out_dir / f"{stem}.rttm"
    txt_path = out_dir / f"{stem}.speaker.txt"
    srt_path = out_dir / f"{stem}.speaker.srt"

    if not force and all(p.exists() and p.stat().st_size > 0 for p in [rttm_path, txt_path, srt_path]):
        logging.info("SKIP diarization exists: %s", asr_wav16k.name)
        return rttm_path, txt_path, srt_path

    whisper_segs = read_whisper_json(whisper_json)
    audio_f32, speech_segs = vad_segments(asr_wav16k, aggressiveness=vad_aggr)
    sr = 16000

    if not speech_segs:
        rttm_path.write_text("", encoding="utf-8")
        txt_path.write_text("", encoding="utf-8")
        srt_path.write_text("", encoding="utf-8")
        return rttm_path, txt_path, srt_path

    encoder = VoiceEncoder()
    embs = []
    seg_times = []
    for (s, e) in speech_segs:
        clip = slice_audio(audio_f32, sr, s, e)
        embs.append(encoder.embed_utterance(clip))
        seg_times.append((s, e))

    embs = np.vstack(embs).astype("float32")
    k = choose_k(embs, 1, max_speakers)
    labels = np.zeros((len(seg_times),), dtype=int) if k == 1 else AgglomerativeClustering(
        n_clusters=k, metric="cosine", linkage="average"
    ).fit_predict(embs)

    diar_segs = [DiarSeg(s, e, f"SPK{lab+1}") for (s, e), lab in zip(seg_times, labels)]
    merged = []
    for ds in diar_segs:
        if not merged: merged.append(ds); continue
        prev = merged[-1]
        if ds.spk == prev.spk and ds.start - prev.end <= 0.25:
            merged[-1] = DiarSeg(prev.start, ds.end, prev.spk)
        else:
            merged.append(ds)

    # RTTM
    file_id = stem
    rttm_lines = []
    for ds in merged:
        dur = max(0.0, ds.end - ds.start)
        rttm_lines.append(f"SPEAKER {file_id} 1 {ds.start:.3f} {dur:.3f} <NA> <NA> {ds.spk} <NA> <NA>")
    rttm_path.write_text("\n".join(rttm_lines) + ("\n" if rttm_lines else ""), encoding="utf-8")

    def speaker_for_window(a: float, b: float) -> str:
        best, bestov = "SPK?", 0.0
        for ds in merged:
            ov = max(0.0, min(b, ds.end) - max(a, ds.start))
            if ov > bestov:
                bestov = ov
                best = ds.spk
        return best

    def ts_srt(sec: float) -> str:
        ms = int(round(sec * 1000.0))
        hh = ms // 3_600_000
        ms -= hh * 3_600_000
        mm = ms // 60_000
        ms -= mm * 60_000
        ss = ms // 1_000
        ms -= ss * 1_000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    txt_lines = []
    srt_lines = []
    idx = 1
    for ws in whisper_segs:
        spk = speaker_for_window(ws.start, ws.end)
        txt_lines.append(f"[{spk}] {ws.text}")
        srt_lines.append(str(idx))
        srt_lines.append(f"{ts_srt(ws.start)} --> {ts_srt(ws.end)}")
        srt_lines.append(f"[{spk}] {ws.text}")
        srt_lines.append("")
        idx += 1

    txt_path.write_text("\n".join(txt_lines) + ("\n" if txt_lines else ""), encoding="utf-8")
    srt_path.write_text("\n".join(srt_lines).rstrip() + ("\n" if srt_lines else ""), encoding="utf-8")

    return rttm_path, txt_path, srt_path
