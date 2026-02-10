# audio-pipeline (profile + config driven)

Single-command, resume-safe pipeline for:

1) Enhancing room recordings (iPhone, back-of-room) using ffmpeg filters  
2) Transcribing with faster-whisper (CPU-friendly with INT8)  
3) Optional token-free diarization (speaker labels) using VAD + embeddings + clustering  

The pipeline now resolves a single configuration object from profile + optional config file + CLI overrides.

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

## Profiles
- `aggressive` (default; matches existing behavior closest)
- `balanced`
- `gentle`

## Run examples
Use aggressive profile:
```bash
audio-pipeline run in_dir out_dir --profile aggressive
```

Balanced profile with diarization disabled:
```bash
audio-pipeline run in_dir out_dir --profile balanced --no-diarize
```

Load config file (YAML or TOML):
```bash
audio-pipeline run in_dir out_dir --config examples/pipeline.yml
```

Dry-run to inspect resolved config and stages:
```bash
audio-pipeline run in_dir out_dir --dry-run
```

## Feature toggles
Enable/disable stages cleanly:
- `--no-enhance`
- `--no-transcribe`
- `--diarize` / `--no-diarize`
- `--no-subtitles`
- `--no-master-outputs`

## Precedence rules
Configuration resolution is (lowest to highest):
1. base defaults
2. selected profile (`--profile`)
3. config file (`--config`)
4. CLI flags

CLI always wins.

## Optional diarization dependencies
Diarization dependencies are imported lazily only when diarization is enabled. If missing and diarization is requested, a clear error is emitted:

> Diarization enabled but dependencies not installed. Install with: pip install -e '.[diarization]'

## Outputs
```
output_root/
  enhanced/            # enhanced .m4a files (if enabled)
  asr_wav/             # 16k mono wav used for ASR/diarization (if enabled)
  transcripts/         # .txt .json and optional .srt/.vtt (+ speaker outputs if diarize)
  pipeline.log
  master_transcripts.csv      # if master outputs enabled
  master_transcript.txt       # if master outputs enabled
```

## Notes
- Originals are never modified.
- If `loudnorm` is missing, it falls back to dynaudnorm (+ alimiter if available).
- For less aggressive output, use `--profile balanced` or `--profile gentle`.
