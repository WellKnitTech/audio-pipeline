from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config_loader import config_pretty_json, resolve_config
from .profiles import PROFILES
from .video import merge_chunk_transcripts, prepare_video
from .song_segmentation import render_song_cuts, scan_video_events


def main() -> int:
    ap = argparse.ArgumentParser(prog="audio-pipeline", description="Batch enhance + transcribe + optional diarization.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="Run the full pipeline on a folder")
    runp.add_argument("input_dir", type=Path, help="Folder containing input audio files")
    runp.add_argument("output_root", type=Path, help="Output root directory")

    runp.add_argument("--profile", choices=sorted(PROFILES), help="Config profile to apply before file/CLI overrides")
    runp.add_argument("--config", type=Path, help="Optional YAML/TOML config file")
    runp.add_argument("--dry-run", action="store_true", help="Print resolved config and enabled stages, then exit")

    runp.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    runp.add_argument("--no-mono", action="store_true", help="Keep stereo (default: downmix to mono)")
    runp.add_argument("--make-asr-wav", action="store_true", help="Create 16k mono wav copies")

    runp.add_argument("--model", default=None, help="Whisper model: tiny, base, small, medium, large-v2, large-v3")
    runp.add_argument("--device", default=None, choices=["cpu", "cuda"], help="cpu or cuda (NVIDIA only)")
    runp.add_argument("--compute-type", default=None, help="cpu compute type: int8 (fast), int8_float16, float32")
    runp.add_argument("--beam-size", type=int, default=None, help="Beam size (accuracy vs speed)")
    runp.add_argument("--language", default=None, help="Force language (e.g. en), otherwise auto-detect")
    runp.add_argument("--vad", action="store_true", help="Enable VAD filter")
    runp.add_argument("--vad-min-silence-ms", type=int, default=None, help="VAD min silence duration")
    runp.add_argument("--temperature", type=float, default=None, help="Decoding temperature (0.0 deterministic)")

    runp.add_argument("--diarize", action="store_true", help="Enable token-free diarization (requires extra deps)")
    runp.add_argument("--max-speakers", type=int, default=None, help="Max speakers to consider")
    runp.add_argument("--vad-aggr", type=int, default=None, choices=[0, 1, 2, 3], help="Diarization VAD aggressiveness")

    runp.add_argument("--no-enhance", action="store_true", help="Disable enhancement stage")
    runp.add_argument("--no-transcribe", action="store_true", help="Disable transcription stage")
    runp.add_argument("--no-diarize", action="store_true", help="Disable diarization stage")
    runp.add_argument("--no-subtitles", action="store_true", help="Disable SRT/VTT subtitle outputs")
    runp.add_argument("--no-master-outputs", action="store_true", help="Disable master transcript aggregation outputs")

    runp.add_argument("--verbose", action="store_true", help="Verbose logging")

    preparep = sub.add_parser("video-prepare", help="Probe a video and extract bounded audio chunks")
    preparep.add_argument("video_path", type=Path, help="Source video file (left unchanged)")
    preparep.add_argument("output_dir", type=Path, help="New output directory for chunks and manifest")
    preparep.add_argument("--chunk-seconds", type=int, default=600, help="Maximum audio chunk duration (default: 600)")

    mergep = sub.add_parser("video-merge", help="Rebase chunk transcripts onto the full video timeline")
    mergep.add_argument("manifest", type=Path, help="video-prepare manifest.json")
    mergep.add_argument("transcript_dir", type=Path, help="Directory containing per-chunk transcript JSON files")
    mergep.add_argument("output_json", type=Path, help="New output JSON path for the merged transcript")

    scanp = sub.add_parser("video-song-scan", help="Scan a concert soundtrack for candidate applause/song events")
    scanp.add_argument("video_path", type=Path, help="Source video; never modified")
    scanp.add_argument("model_path", type=Path, help="YAMNet ONNX model file")
    scanp.add_argument("output_csv", type=Path, help="New CSV path for per-frame event scores")
    scanp.add_argument("--chunk-seconds", type=int, default=30, help="Inference chunk target; aligned upward to the YAMNet frame hop")
    scanp.add_argument("--threshold", type=float, default=0.18, help="Crowd-response score threshold")
    scanp.add_argument("--merge-gap", type=float, default=3.0, help="Merge candidate frames separated by at most this many seconds")
    scanp.add_argument("--min-duration", type=float, default=1.5, help="Minimum candidate applause duration")
    scanp.add_argument("--vocal-ceiling", type=float, default=0.25, help="Exclude frames with singing/choir score at or above this value")

    renderp = sub.add_parser("video-song-render", help="Render reviewed song ranges from a JSON cue sheet")
    renderp.add_argument("video_path", type=Path, help="Source video; never modified")
    renderp.add_argument("cut_list", type=Path, help="JSON object with a 'cuts' array or a cut array")
    renderp.add_argument("output_dir", type=Path, help="New output directory for clips and manifest")
    renderp.add_argument("--height", type=int, default=720, help="Output height (default: 720; aspect ratio preserved)")
    renderp.add_argument("--max-size-mib", type=float, default=95.0, help="Maximum file size per clip (default: 95 MiB)")
    renderp.add_argument("--fade-seconds", type=float, default=3.0, help="Audio fade length at each clip end")
    renderp.add_argument("--encoder", default="h264_nvenc", help="FFmpeg video encoder (default: h264_nvenc)")
    renderp.add_argument("--preset", help="Encoder preset (default: p5 for NVENC, medium otherwise)")
    renderp.add_argument("--cardwire-gpu", help="GPU ID to pass to 'cardwire launch --gpu' (for Cardwire Smart mode)")
    renderp.add_argument("--audio-bitrate", type=int, default=128_000, help="AAC audio bitrate in bits/second")

    args = ap.parse_args()

    if args.cmd == "run":
        from .pipeline import enabled_stage_names, run_pipeline

        if not args.input_dir.is_dir():
            ap.error(f"input_dir is not a directory: {args.input_dir}")

        cli_overrides = [
            ("force", True if args.force else None),
            ("mono", False if args.no_mono else None),
            ("features.make_asr_wav", True if args.make_asr_wav else None),
            ("features.enhance", False if args.no_enhance else None),
            ("features.transcribe", False if args.no_transcribe else None),
            ("features.diarize", True if args.diarize else None),
            ("features.diarize", False if args.no_diarize else None),
            ("features.subtitles", False if args.no_subtitles else None),
            ("features.master_outputs", False if args.no_master_outputs else None),
            ("asr.model", args.model),
            ("asr.device", args.device),
            ("asr.compute_type", args.compute_type),
            ("asr.beam_size", args.beam_size),
            ("asr.language", args.language),
            ("asr.vad", True if args.vad else None),
            ("asr.vad_min_silence_ms", args.vad_min_silence_ms),
            ("asr.temperature", args.temperature),
            ("diarization.max_speakers", args.max_speakers),
            ("diarization.vad_aggr", args.vad_aggr),
        ]

        try:
            config = resolve_config(
                profile_name=args.profile,
                config_path=args.config,
                cli_overrides=cli_overrides,
            )
        except Exception as exc:
            ap.error(str(exc))

        if args.dry_run:
            print("Resolved config:")
            print(config_pretty_json(config))
            print("Enabled stages:", ", ".join(enabled_stage_names(config)) or "none")
            return 0

        log_path = args.output_root / "pipeline.log"
        args.output_root.mkdir(parents=True, exist_ok=True)

        level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

        logging.info("Input:  %s", args.input_dir)
        logging.info("Output: %s", args.output_root)
        logging.info("Stages: %s", ", ".join(enabled_stage_names(config)) or "none")

        run_pipeline(
            input_dir=args.input_dir,
            output_root=args.output_root,
            config=config,
        )

        logging.info("Done.")
        return 0

    if args.cmd == "video-prepare":
        try:
            manifest = prepare_video(args.video_path, args.output_dir, chunk_seconds=args.chunk_seconds)
        except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
            ap.error(str(exc))
        print(json.dumps({
            "output_dir": str(args.output_dir),
            "duration_seconds": manifest["duration_seconds"],
            "audio_chunks": len(manifest["chunks"]),
            "manifest": str(args.output_dir / "manifest.json"),
        }, indent=2))
        return 0

    if args.cmd == "video-merge":
        try:
            result = merge_chunk_transcripts(args.manifest, args.transcript_dir, args.output_json)
        except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
            ap.error(str(exc))
        print(json.dumps({
            "output_json": str(args.output_json),
            "chunks": result["meta"]["chunk_count"],
            "segments": len(result["segments"]),
        }, indent=2))
        return 0

    if args.cmd == "video-song-scan":
        try:
            metadata = scan_video_events(
                args.video_path, args.model_path, args.output_csv,
                chunk_seconds=args.chunk_seconds, threshold=args.threshold, merge_gap_seconds=args.merge_gap,
                min_duration_seconds=args.min_duration, vocal_ceiling=args.vocal_ceiling,
            )
        except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as exc:
            ap.error(str(exc))
        print(json.dumps({"output_csv": str(args.output_csv),
                          "metadata": str(args.output_csv.with_suffix('.metadata.json')),
                          "frames": metadata["frame_count"],
                          "candidate_applause_spans": metadata["candidate_applause_spans"]}, indent=2))
        return 0

    if args.cmd == "video-song-render":
        try:
            data = json.loads(args.cut_list.read_text(encoding="utf-8"))
            cuts = data.get("cuts") if isinstance(data, dict) else data
            if not isinstance(cuts, list):
                raise ValueError("cut list JSON must be an array or an object with a 'cuts' array")
            manifest = render_song_cuts(
                args.video_path, cuts, args.output_dir, height=args.height,
                max_size_mib=args.max_size_mib, fade_seconds=args.fade_seconds,
                encoder=args.encoder, preset=args.preset, cardwire_gpu=args.cardwire_gpu,
                audio_bitrate=args.audio_bitrate,
            )
        except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            ap.error(str(exc))
        print(json.dumps({"output_dir": str(args.output_dir), "clips": len(manifest["clips"]),
                          "manifest": str(args.output_dir / "manifest.json")}, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
