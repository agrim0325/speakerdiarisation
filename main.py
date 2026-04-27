"""
Speaker Diarization Pipeline
Python/ML conversion of the MATLAB SAS Speaker Turn Pipeline (G37).

Usage:
    python main.py
    python main.py --mic1 path/to/a1.mp3 --mic2 path/to/a2.mp3
    python main.py --num-speakers 3
"""

import argparse
import os
import sys
from pathlib import Path

from src.pipeline import SpeakerDiarizationPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Speaker Diarization Pipeline (Python ML version of G37 MATLAB project)"
    )
    parser.add_argument("--mic1", type=str, default=None, help="Path to microphone 1 audio file")
    parser.add_argument("--mic2", type=str, default=None, help="Path to microphone 2 audio file")
    parser.add_argument("--mic-dir", type=str, default="Input (Microphone)", help="Directory containing mic files")
    parser.add_argument("--dataset-dir", type=str, default="Input (Dataset)", help="Directory with reference speaker files")
    parser.add_argument("--output-dir", type=str, default="Output", help="Output directory")
    parser.add_argument("--num-speakers", type=int, default=None, help="Force number of speakers (auto-detect if not set)")
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--max-speakers", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--energy-threshold-pct", type=float, default=1.5)
    parser.add_argument("--min-segment-duration", type=float, default=0.20)
    parser.add_argument("--merge-gap", type=float, default=0.50)
    return parser.parse_args()


def main():
    args = parse_args()

    root = Path(__file__).parent

    pipeline = SpeakerDiarizationPipeline(
        mic1_path=args.mic1,
        mic2_path=args.mic2,
        mic_dir=root / args.mic_dir,
        dataset_dir=root / args.dataset_dir,
        output_dir=root / args.output_dir,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        sample_rate=args.sample_rate,
        energy_threshold_pct=args.energy_threshold_pct,
        min_segment_duration=args.min_segment_duration,
        merge_gap=args.merge_gap,
    )

    summary = pipeline.run()
    print("\nDone. Summary written to Output/reports/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
