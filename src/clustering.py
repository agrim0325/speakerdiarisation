"""
Clustering of speech segments using scikit-learn KMeans.
Mirrors cluster_segment_features(), evaluate_speaker_count(),
choose_best_candidate(), and best_unique_assignment() from working.m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .models import SpeakerModel, score_segment_vs_model


@dataclass
class ClusterCandidate:
    num_clusters: int
    labels: np.ndarray              # (N_segments,) cluster label per segment
    score_matrix: np.ndarray        # (k, n_refs) log-likelihood scores
    assignment: dict[int, str]      # cluster_id -> speaker_name
    total_score: float
    mean_score: float
    avg_margin: float
    min_margin: float


def cluster_and_assign(
    segment_features: np.ndarray,
    segment_audio: list[np.ndarray],
    reference_models: list[SpeakerModel],
    sample_rate: int,
    num_speakers: int | None = None,
    min_speakers: int = 2,
    max_speakers: int = 5,
    score_tolerance: float = 0.25,
) -> ClusterCandidate:
    """
    Main entry: cluster segments and assign each cluster to a speaker.

    If num_speakers is None, auto-selects the best k in [min_speakers, max_speakers].
    """
    ref_names = [m.name for m in reference_models]
    max_k = min(max_speakers, len(ref_names), len(segment_features))
    min_k = min(min_speakers, max_k)

    if num_speakers is not None:
        k = max(1, min(num_speakers, max_k))
        print(f"  Forced speaker count: {k}")
        return _evaluate_k(k, segment_features, segment_audio, reference_models, sample_rate)

    # Auto-select: evaluate every k in range
    candidates: list[ClusterCandidate] = []
    for k in range(min_k, max_k + 1):
        c = _evaluate_k(k, segment_features, segment_audio, reference_models, sample_rate)
        candidates.append(c)
        print(f"  k={k}: mean_score={c.mean_score:.3f}  avg_margin={c.avg_margin:.3f}")

    return _choose_best(candidates, score_tolerance)


def _evaluate_k(
    k: int,
    features: np.ndarray,
    segment_audio: list[np.ndarray],
    reference_models: list[SpeakerModel],
    sample_rate: int,
) -> ClusterCandidate:
    labels = _run_kmeans(features, k)

    # Build per-cluster concatenated audio
    cluster_audio = _build_cluster_audio(segment_audio, labels, k)

    # Score each cluster against each reference model
    score_matrix = np.zeros((k, len(reference_models)))
    margins = np.zeros(k)
    for ci in range(k):
        for ri, model in enumerate(reference_models):
            score_matrix[ci, ri] = score_segment_vs_model(cluster_audio[ci], model, sample_rate)
        sorted_scores = np.sort(score_matrix[ci])[::-1]
        margins[ci] = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else np.inf

    ref_names = [m.name for m in reference_models]
    assignment, total_score = _best_unique_assignment(score_matrix, ref_names)

    return ClusterCandidate(
        num_clusters=k,
        labels=labels,
        score_matrix=score_matrix,
        assignment=assignment,
        total_score=total_score,
        mean_score=total_score / max(k, 1),
        avg_margin=float(np.mean(margins)),
        min_margin=float(np.min(margins)),
    )


def _run_kmeans(features: np.ndarray, k: int) -> np.ndarray:
    """Z-score normalise then run KMeans++ (12 inits, 60 max_iter)."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=12,
        max_iter=60,
        random_state=42,
    )
    labels = km.fit_predict(scaled) + 1  # 1-indexed to match MATLAB
    return labels


def _build_cluster_audio(
    segment_audio: list[np.ndarray], labels: np.ndarray, k: int
) -> list[np.ndarray]:
    from .audio import normalize_signal
    cluster_audio = []
    for ci in range(1, k + 1):
        indices = np.where(labels == ci)[0]
        if len(indices) == 0:
            cluster_audio.append(np.zeros(1600))  # 0.1 s silence
        else:
            combined = np.concatenate([segment_audio[i].ravel() for i in indices])
            cluster_audio.append(normalize_signal(combined))
    return cluster_audio


def _best_unique_assignment(
    score_matrix: np.ndarray, ref_names: list[str]
) -> tuple[dict[int, str], float]:
    """
    Exhaustive Hungarian-style assignment (brute-force over permutations).
    Mirrors best_unique_assignment() in working.m.
    """
    n_clusters = score_matrix.shape[0]
    ref_ids = list(range(len(ref_names)))
    best_total = -np.inf
    best_assignment: dict[int, str] = {}

    for perm in permutations(ref_ids, n_clusters):
        total = sum(score_matrix[ci, perm[ci]] for ci in range(n_clusters))
        if total > best_total:
            best_total = total
            best_assignment = {ci + 1: ref_names[perm[ci]] for ci in range(n_clusters)}

    return best_assignment, best_total


def _choose_best(candidates: list[ClusterCandidate], tolerance: float) -> ClusterCandidate:
    """
    Select the best k: among those within `tolerance` of the highest mean score,
    prefer higher k, then higher avg_margin, then higher mean_score.
    """
    best_mean = max(c.mean_score for c in candidates)
    near = [c for c in candidates if (best_mean - c.mean_score) <= tolerance]
    near.sort(key=lambda c: (c.num_clusters, c.avg_margin, c.mean_score), reverse=True)
    chosen = near[0]
    print(f"  Selected: k={chosen.num_clusters}")
    return chosen
