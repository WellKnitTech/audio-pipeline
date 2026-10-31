# audio-pipeline (profile + config driven)

Single-command, resume-safe pipeline for:

1) Enhancing room recordings (iPhone, back-of-room) using ffmpeg filters  
2) Transcribing with faster-whisper (CPU-friendly with INT8)  
3) Optional token-free diarization (speaker labels) using VAD + embeddings + clustering  
4) Song-by-song concert video segmentation using reviewed cut ranges and YAMNet event candidates

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

## Song-by-song concert segmentation

The optional YAMNet workflow scans the full audio for candidate applause and
vocal activity, but **does not auto-pick final cuts**. Review the event CSV and
listen to ambiguous boundaries; choir clapping and cheering can look like an
audience response. Put approved ranges in a JSON cut list based on
`examples/song-cuts.template.json`. `examples/centennial-candidate-cuts.json`
records the candidate times used for the source recording below; it is not a
fully audited cut list (only Song 02 was adjusted after listening feedback).
Review all boundaries before treating those ranges as final.

Then acquire a source recording you have permission to download. For YouTube, inspect available formats first and avoid playlist expansion:

```bash
yt-dlp --no-playlist -F 'https://www.youtube.com/watch?v=VIDEO_ID'
yt-dlp --no-playlist -f 'bv*[height<=1080]+ba/b[height<=1080]' \\
  --merge-output-format mp4 -o 'recording.%(ext)s' 'https://www.youtube.com/watch?v=VIDEO_ID'
```

Install the optional inference dependencies and FFmpeg:

```bash
pip install -e ".[song-segmentation]"
# Install ffmpeg and ffprobe using your OS package manager.
```

YAMNet model files are third-party artifacts and are not included in this
repository. This pinned Hugging Face revision matches the ONNX model used here;
the checksum makes the download auditable. Check the model and AudioSet licensing
terms before redistribution or commercial use.

```bash
mkdir -p models
curl -L 'https://huggingface.co/jafet21/yamnetonnx/resolve/ebb48ac2fc8c587f8baa7ae7f9ba34b2ee55d8f3/yamnet.onnx' -o models/yamnet.onnx
printf '%s  %s\n' '04e27fca08e7a3aea2630d1a63a51e6b437c803e0ff6e26399f80870ac251dda' 'models/yamnet.onnx' | sha256sum --check
```

Scan and review candidates:

```bash
audio-pipeline video-song-scan recording.mp4 models/yamnet.onnx work/events.csv
```

The scanner writes per-frame YAMNet class scores to CSV and writes a companion
`.metadata.json` with source/model SHA-256 hashes, Python/NumPy/ONNX Runtime/FFmpeg
versions, class IDs, scan settings, stream start offset, and candidate spans.
Times use the first video stream as zero, matching the render command; chunks are
aligned to the model's 0.48-second frame grid and retain overlap at boundaries.
Its defaults suppress clapping while singing/choir scores remain high. These are
navigation aids, not authoritative boundaries.

Create a reviewed cut list and render it. Range times are seconds relative to the
first video frame (the same origin as scanner output); keep entries ordered and
non-overlapping. Gaps are allowed (for example, an intermission). The renderer
preserves aspect ratio, fades the final 3 seconds of audio, targets a conservative
per-clip size, full-decodes each output, and refuses to overwrite files/directories.

```bash
audio-pipeline video-song-render recording.mp4 cuts.json output/songs \\
  --height 720 --max-size-mib 95 --cardwire-gpu 1
```

`--cardwire-gpu ID` wraps FFmpeg with `cardwire launch --gpu ID`, useful in
Cardwire Smart mode; find the ID with `cardwire list`. NVIDIA NVENC is the
renderer default. If unavailable, explicitly choose `--encoder libx264`; software
encoding is never selected silently. Each run writes a `manifest.json` with
source, source hash, cut times, Python/FFmpeg versions, encoder, preset, size,
bitrate, fade, and boundary notes. The source video and audio are never modified.

The renderer checks each final file against the size target and fails rather than
leaving an oversized export. Lower `--max-size-mib` or split a long cut if the
destination has a stricter attachment cap.

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
