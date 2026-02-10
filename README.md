# audio-pipeline (aggressive loudness build)

Single-command, resume-safe pipeline for:

1) Enhancing room recordings (iPhone, back-of-room) using ffmpeg filters  
2) Transcribing with faster-whisper (CPU-friendly with INT8)  
3) Optional token-free diarization (speaker labels) using VAD + embeddings + clustering  

This build is tuned to be **more aggressive on volume**:
- Compressor: threshold -20 dB, ratio 4:1, makeup +12 dB
- Loudness target: -12 LUFS
- True peak cap: -1.0 dB

## Requirements
- Linux
- `ffmpeg` installed and in PATH
- Python 3.9+

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional diarization (no token required):
```bash
pip install -e ".[diarization]"
```

## Run
Enhance + Transcribe (recommended defaults for Intel CPU):
```bash
audio-pipeline run /path/to/input_audio /path/to/output_root \
  --model medium --device cpu --compute-type int8 --vad --beam-size 7 --make-asr-wav
```

Add diarization:
```bash
audio-pipeline run /path/to/input_audio /path/to/output_root \
  --model medium --device cpu --compute-type int8 --vad --beam-size 7 --make-asr-wav \
  --diarize --max-speakers 6 --vad-aggr 2
```

Outputs:
```
output_root/
  enhanced/            # enhanced .m4a files
  asr_wav/             # 16k mono wav used for ASR/diarization
  transcripts/         # .txt .srt .vtt .json per file (+ speaker outputs if diarize)
  pipeline.log
  master_transcripts.csv
  master_transcript.txt
```

## Notes
- Originals are never modified.
- If `loudnorm` is missing, it falls back to dynaudnorm (+ alimiter if available). The fallback is also loud.
- If the result is too harsh/pumpy, edit `audio_pipeline/enhance.py` and reduce makeup to 10 or set loudnorm I=-13/-14.
