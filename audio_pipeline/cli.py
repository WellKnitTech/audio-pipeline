from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import run_pipeline

def main() -> int:
    ap = argparse.ArgumentParser(prog="audio-pipeline", description="Batch enhance + transcribe + optional diarization.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="Run the full pipeline on a folder")
    runp.add_argument("input_dir", type=Path, help="Folder containing input audio files")
    runp.add_argument("output_root", type=Path, help="Output root directory")

    runp.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    runp.add_argument("--no-mono", action="store_true", help="Keep stereo (default: downmix to mono)")
    runp.add_argument("--make-asr-wav", action="store_true", help="Create 16k mono wav copies (recommended for consistency)")

    runp.add_argument("--model", default="medium", help="Whisper model: tiny, base, small, medium, large-v2, large-v3")
    runp.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cpu or cuda (NVIDIA only)")
    runp.add_argument("--compute-type", default="int8", help="cpu compute type: int8 (fast), int8_float16, float32")
    runp.add_argument("--beam-size", type=int, default=7, help="Beam size (accuracy vs speed)")
    runp.add_argument("--language", default=None, help="Force language (e.g. en), otherwise auto-detect")
    runp.add_argument("--vad", action="store_true", help="Enable VAD filter (recommended for room recordings)")
    runp.add_argument("--vad-min-silence-ms", type=int, default=500, help="VAD min silence duration")
    runp.add_argument("--temperature", type=float, default=0.0, help="Decoding temperature (0.0 deterministic)")

    runp.add_argument("--diarize", action="store_true", help="Enable token-free diarization (requires extra deps)")
    runp.add_argument("--max-speakers", type=int, default=6, help="Max speakers to consider")
    runp.add_argument("--vad-aggr", type=int, default=2, choices=[0,1,2,3], help="Diarization VAD aggressiveness")

    runp.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = ap.parse_args()

    if args.cmd == "run":
        if not args.input_dir.is_dir():
            ap.error(f"input_dir is not a directory: {args.input_dir}")

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

        run_pipeline(
            input_dir=args.input_dir,
            output_root=args.output_root,
            force=args.force,
            mono=not args.no_mono,
            make_asr_wav=args.make_asr_wav,
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            language=args.language,
            vad=args.vad,
            vad_min_silence_ms=args.vad_min_silence_ms,
            temperature=args.temperature,
            diarize=args.diarize,
            max_speakers=args.max_speakers,
            vad_aggr=args.vad_aggr,
        )

        logging.info("Done.")
        return 0

    return 2

if __name__ == "__main__":
    raise SystemExit(main())
