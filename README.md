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

## Prepare a large video for transcription

`video-prepare` probes a local video, leaves the source untouched, and extracts
16 kHz mono PCM audio into bounded WAV chunks. The manifest records source
metadata and chunk offsets relative to the first video frame, including any
audio/video stream start-time difference.

```bash
audio-pipeline video-prepare /path/to/event.mp4 /path/to/work/event --chunk-seconds 600
```

The chunks can be transcribed with the existing pipeline, then merged onto the
full video timeline:

The run command writes transcript files beneath its output root's `transcripts/`
subdirectory; pass that directory to the merge step:

```bash
audio-pipeline run /path/to/work/event/audio /path/to/work/transcribe-run --no-enhance
audio-pipeline video-merge /path/to/work/event/manifest.json /path/to/work/transcribe-run/transcripts /path/to/work/event-transcript.json
```

`video-merge` requires one transcript JSON per chunk and writes global timestamps
plus chunk provenance. It accepts the transcript naming used by regular and
ASR-WAV runs. Speaker identities are not yet reconciled across chunks; highlight
selection and video clip rendering remain follow-up work.

Preparation and merging leave source inputs untouched and refuse to overwrite
outputs.

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
- The CLI now shows tqdm progress bars for total files and per-file stages in addition to log lines.
- Originals are never modified.
- If `loudnorm` is missing, it falls back to dynaudnorm (+ alimiter if available).
- For less aggressive output, use `--profile balanced` or `--profile gentle`.
