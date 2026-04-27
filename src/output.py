"""
Output generation: turn clips, speaker clips, CSV/XLSX/JSON reports, plots.
Mirrors write_turn_outputs(), render_waveform_plot(), render_cluster_plot() from working.m.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .audio import fade_edges, save_audio, normalize_signal
from .clustering import ClusterCandidate


@dataclass
class TurnRow:
    TurnID: int
    FileName: str
    Speaker: str
    StartTimeS: float
    EndTimeS: float
    ClipDurationS: float
    ClusterID: int
    BestMatchName: str
    MatchScore: float
    MatchMargin: float
    LogRmsRatio: float
    DelayMs: float
    SpectralCentroidKHz: float


@dataclass
class SpeakerRow:
    Speaker: str
    FileName: str
    SpeechDurationS: float
    Turns: int


def write_outputs(
    output_dir: Path,
    segments: np.ndarray,
    candidate: ClusterCandidate,
    segment_features: np.ndarray,
    segment_audio: list[np.ndarray],
    mic1: np.ndarray,
    mic2: np.ndarray,
    sample_rate: int,
    mic1_path: Path,
    mic2_path: Path,
    combined_path: Path,
    ref_names: list[str],
) -> dict[str, Any]:
    """Write all outputs and return a summary dict."""

    turns_dir = output_dir / "turn_clips"
    speakers_dir = output_dir / "speaker_clips"
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    for d in [turns_dir, speakers_dir, reports_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    turn_rows: list[TurnRow] = []
    speaker_audio_map: dict[str, list[np.ndarray]] = {}

    score_matrix = candidate.score_matrix
    assignment = candidate.assignment  # {cluster_id (1-indexed) -> speaker_name}

    for i, (t_start, t_end) in enumerate(segments):
        cluster_id = int(candidate.labels[i])
        speaker_name = assignment[cluster_id]
        safe_name = _safe_stem(speaker_name)
        file_name = f"turn_{i + 1:02d}_{safe_name}.wav"
        clip_path = turns_dir / file_name

        clip = fade_edges(segment_audio[i], sample_rate)
        save_audio(clip_path, clip, sample_rate)

        speaker_audio_map.setdefault(speaker_name, []).append(clip)

        cluster_scores = score_matrix[cluster_id - 1]
        sorted_idx = np.argsort(cluster_scores)[::-1]
        best_name = ref_names[sorted_idx[0]] if len(ref_names) > 0 else speaker_name
        best_score = float(cluster_scores[sorted_idx[0]])
        margin = float(cluster_scores[sorted_idx[0]] - cluster_scores[sorted_idx[1]]) if len(sorted_idx) > 1 else np.inf

        turn_rows.append(TurnRow(
            TurnID=i + 1,
            FileName=file_name,
            Speaker=speaker_name,
            StartTimeS=round(t_start, 3),
            EndTimeS=round(t_end, 3),
            ClipDurationS=round(t_end - t_start, 3),
            ClusterID=cluster_id,
            BestMatchName=best_name,
            MatchScore=round(best_score, 3),
            MatchMargin=round(margin, 3),
            LogRmsRatio=round(float(segment_features[i, 0]), 5),
            DelayMs=round(float(segment_features[i, 2]), 5),
            SpectralCentroidKHz=round(float(segment_features[i, 4]), 5),
        ))

    # Speaker clips
    speaker_rows: list[SpeakerRow] = []
    for speaker_name in sorted(speaker_audio_map):
        chunks = speaker_audio_map[speaker_name]
        merged = normalize_signal(np.concatenate([c.ravel() for c in chunks]))
        file_name = f"{_safe_stem(speaker_name)}_all_turns.wav"
        save_audio(speakers_dir / file_name, merged, sample_rate)
        speaker_rows.append(SpeakerRow(
            Speaker=speaker_name,
            FileName=file_name,
            SpeechDurationS=round(len(merged) / sample_rate, 3),
            Turns=len(chunks),
        ))

    # CSV
    csv_path = reports_dir / "turns.csv"
    pd.DataFrame([asdict(r) for r in turn_rows]).to_csv(csv_path, index=False)
    print(f"  CSV:   {csv_path}")

    # XLSX
    xlsx_path = output_dir / "speaker_timeline.xlsx"
    try:
        df = pd.DataFrame([asdict(r) for r in turn_rows])
        df.to_excel(xlsx_path, index=False)
        xlsx_ok = True
        print(f"  XLSX:  {xlsx_path}")
    except Exception as exc:
        xlsx_ok = False
        print(f"  XLSX export failed ({exc}); CSV still written.")

    # Plots
    active_names = [r.Speaker for r in speaker_rows]
    inactive = [n for n in ref_names if n not in active_names]

    _plot_waveforms(
        plots_dir / "waveforms_labelled.png",
        mic1, mic2, segments, candidate, sample_rate,
    )
    _plot_clusters(
        plots_dir / "speaker_clusters.png",
        candidate,
    )
    print(f"  Plots: {plots_dir}")

    # Summary JSON
    summary = {
        "pipeline": "python_ml_speaker_diarization",
        "ml_techniques_used": [
            "MFCC feature extraction (librosa)",
            "K-Means++ clustering (scikit-learn)",
            "Diagonal Gaussian speaker models",
            "Short-time energy VAD with Hamming windowing",
            "GCC-PHAT cross-correlation delay estimation",
            "Spectral centroid, rolloff, ZCR, flatness features",
            "pyin pitch estimation",
            "Optimal cluster-to-speaker assignment via permutation search",
        ],
        "inputs": {
            "mic1": str(mic1_path),
            "mic2": str(mic2_path),
            "reference_speakers": ref_names,
        },
        "outputs": {
            "combined_wav": str(combined_path),
            "turn_count": len(turn_rows),
            "speaker_count_selected": candidate.num_clusters,
            "active_speakers": active_names,
            "inactive_dataset_speakers": inactive,
            "xlsx_written": xlsx_ok,
            "total_speech_duration_s": round(sum(r.ClipDurationS for r in turn_rows), 3),
        },
        "selection_metrics": {
            "mean_assignment_score": round(candidate.mean_score, 4),
            "avg_margin": round(candidate.avg_margin, 4),
            "min_margin": round(candidate.min_margin, 4),
        },
        "turns": [asdict(r) for r in turn_rows],
        "speaker_clips": [asdict(r) for r in speaker_rows],
    }
    json_path = reports_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON:  {json_path}")

    return summary


# ── Plotting ──────────────────────────────────────────────────────────────────

_COLORS = plt.cm.tab10.colors


def _plot_waveforms(
    out_path: Path,
    mic1: np.ndarray,
    mic2: np.ndarray,
    segments: np.ndarray,
    candidate: ClusterCandidate,
    sample_rate: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="white")
    time1 = np.arange(len(mic1)) / sample_rate
    time2 = np.arange(len(mic2)) / sample_rate

    for ax, sig, time, title in [
        (axes[0], mic1, time1, "Mic 1 – Labelled Turns"),
        (axes[1], mic2, time2, "Mic 2 – Labelled Turns"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-1.1, 1.1)
        ax.plot(time, sig, color="black", linewidth=0.5, alpha=0.8)

        for i, (t_s, t_e) in enumerate(segments):
            cid = int(candidate.labels[i])
            speaker = candidate.assignment[cid]
            color = _COLORS[(cid - 1) % len(_COLORS)]
            ax.axvspan(t_s, t_e, alpha=0.15, color=color)
            ax.text(
                t_s, 0.85 - 0.12 * (i % 5),
                speaker, color=color, fontsize=7, clip_on=True,
            )

    # Legend
    handles = [
        mpatches.Patch(color=_COLORS[(cid - 1) % len(_COLORS)], label=spk)
        for cid, spk in sorted(candidate.assignment.items())
    ]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_clusters(out_path: Path, candidate: ClusterCandidate) -> None:
    """2-D scatter of the first two segment features, coloured by cluster."""
    # We store the full feature matrix in the candidate via score_matrix proxy;
    # use score_matrix rows as 2-D proxy (log-likelihood vs. each reference).
    scores = candidate.score_matrix   # (k, n_refs) – already 2-D, good for scatter

    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor="white")
    ax.set_title("Speaker Clusters (log-likelihood space)")
    ax.set_xlabel("Best match score")
    ax.set_ylabel("Second best score")
    ax.grid(True, linestyle="--", alpha=0.4)

    for ci in range(candidate.num_clusters):
        cid = ci + 1
        spk = candidate.assignment[cid]
        color = _COLORS[ci % len(_COLORS)]
        sorted_s = np.sort(scores[ci])[::-1]
        x = sorted_s[0] if len(sorted_s) > 0 else 0
        y = sorted_s[1] if len(sorted_s) > 1 else 0
        ax.scatter(x, y, s=200, color=color, edgecolors="k", label=spk, zorder=3)
        ax.annotate(spk, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9, color=color)

    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_stem(name: str) -> str:
    import re
    stem = re.sub(r"[^A-Za-z0-9._\-]+", "_", name)
    stem = stem.strip("_.")
    return stem or "speaker"
