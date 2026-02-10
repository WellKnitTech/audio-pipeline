from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config_loader import config_pretty_json, resolve_config
from .pipeline import enabled_stage_names, run_pipeline
from .profiles import PROFILES


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

    args = ap.parse_args()

    if args.cmd == "run":
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

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
