"""
Top-level pipeline orchestrator.
Mirrors the main working() function from working.m.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .audio import load_audio, mix_microphones, list_audio_files, save_audio
from .segmentation import detect_segments
from .features import compute_segment_features
from .models import fit_reference_models
from .clustering import cluster_and_assign
from .output import write_outputs


class SpeakerDiarizationPipeline:
    def __init__(
        self,
        mic1_path: str | None = None,
        mic2_path: str | None = None,
        mic_dir: Path = Path("Input (Microphone)"),
        dataset_dir: Path = Path("Input (Dataset)"),
        output_dir: Path = Path("Output"),
        num_speakers: int | None = None,
        min_speakers: int = 2,
        max_speakers: int = 5,
        sample_rate: int = 16000,
        energy_threshold_pct: float = 1.5,
        min_segment_duration: float = 0.20,
        merge_gap: float = 0.50,
        score_tolerance: float = 0.25,
    ):
        self.mic1_path = Path(mic1_path) if mic1_path else None
        self.mic2_path = Path(mic2_path) if mic2_path else None
        self.mic_dir = Path(mic_dir)
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.sample_rate = sample_rate
        self.energy_threshold_pct = energy_threshold_pct
        self.min_segment_duration = min_segment_duration
        self.merge_gap = merge_gap
        self.score_tolerance = score_tolerance

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Resolve inputs ────────────────────────────────────────────────────
        mic1_path, mic2_path = self._resolve_mic_paths()
        ref_paths = list_audio_files(self.dataset_dir)
        if not ref_paths:
            raise FileNotFoundError(f"No reference speaker files found in {self.dataset_dir}")

        print("=" * 72)
        print("Python ML Speaker Diarization Pipeline")
        print(f"Mic 1        : {mic1_path}")
        print(f"Mic 2        : {mic2_path}")
        print(f"Dataset dir  : {self.dataset_dir}")
        print(f"Output dir   : {self.output_dir}")
        print("=" * 72)

        # ── 1. Load & mix ─────────────────────────────────────────────────────
        print("\n[1/7] Loading and mixing audio files")
        mic1 = load_audio(mic1_path, self.sample_rate)
        mic2 = load_audio(mic2_path, self.sample_rate)
        length = min(len(mic1), len(mic2))
        mic1, mic2 = mic1[:length], mic2[:length]
        combined = mix_microphones(mic1, mic2)
        combined_path = self.output_dir / "combined.wav"
        save_audio(combined_path, combined, self.sample_rate)
        print(f"  Duration   : {len(combined) / self.sample_rate:.2f}s")

        # ── 2. VAD segmentation ───────────────────────────────────────────────
        print("\n[2/7] Detecting speech turns (short-time energy VAD)")
        segments = detect_segments(
            combined,
            self.sample_rate,
            self.energy_threshold_pct,
            self.min_segment_duration,
            self.merge_gap,
        )
        if len(segments) == 0:
            raise RuntimeError("No speech turns detected. Try lowering --energy-threshold-pct.")
        print(f"  Turns found: {len(segments)}")

        # ── 3. Segment features ───────────────────────────────────────────────
        print("\n[3/7] Computing segment features (MFCC + spectral + mic delay)")
        seg_features, seg_audio = compute_segment_features(mic1, mic2, segments, self.sample_rate)

        # ── 4. Reference models ───────────────────────────────────────────────
        print("\n[4/7] Building reference speaker models (diagonal Gaussian)")
        ref_models = fit_reference_models(ref_paths, self.sample_rate)
        ref_names = [m.name for m in ref_models]

        # ── 5. Cluster & assign ───────────────────────────────────────────────
        print("\n[5/7] Clustering segments and assigning speakers (KMeans++)")
        candidate = cluster_and_assign(
            seg_features,
            seg_audio,
            ref_models,
            self.sample_rate,
            num_speakers=self.num_speakers,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            score_tolerance=self.score_tolerance,
        )

        # ── 6. Write outputs ──────────────────────────────────────────────────
        print("\n[6/7] Writing turn clips, speaker clips, and reports")
        summary = write_outputs(
            self.output_dir,
            segments,
            candidate,
            seg_features,
            seg_audio,
            mic1, mic2,
            self.sample_rate,
            mic1_path, mic2_path,
            combined_path,
            ref_names,
        )

        # ── 7. Done ───────────────────────────────────────────────────────────
        print("\n[7/7] Complete")
        active = summary["outputs"]["active_speakers"]
        inactive = summary["outputs"]["inactive_dataset_speakers"]
        print(f"  Turns         : {summary['outputs']['turn_count']}")
        print(f"  Active spkrs  : {', '.join(active)}")
        if inactive:
            print(f"  Not detected  : {', '.join(inactive)}")

        return summary

    def _resolve_mic_paths(self) -> tuple[Path, Path]:
        if self.mic1_path and self.mic2_path:
            return self.mic1_path, self.mic2_path

        files = list_audio_files(self.mic_dir)
        if len(files) < 2:
            raise FileNotFoundError(
                f"Expected at least 2 mic files in {self.mic_dir}, found {len(files)}."
            )

        def _rank(p: Path) -> int:
            n = p.stem.lower()
            if any(k in n for k in ("mic1", "a1", "left")):
                return 0
            if any(k in n for k in ("mic2", "a2", "right")):
                return 1
            return 2

        files.sort(key=_rank)
        return files[0], files[1]
