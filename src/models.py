"""
Speaker reference models using Diagonal Gaussian (GMM with 1 component).
Mirrors fit_reference_models() and average_log_likelihood() from working.m.
Also wraps sklearn's GaussianMixture for a richer optional path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import librosa

from .audio import load_audio
from .features import extract_speaker_embedding, N_MFCC, N_FFT, HOP_LENGTH, N_MELS


@dataclass
class SpeakerModel:
    name: str
    mean: np.ndarray       # (D,) mean of MFCC frame matrix
    var: np.ndarray        # (D,) variance + floor


def fit_reference_models(
    reference_paths: list[Path],
    sample_rate: int,
) -> list[SpeakerModel]:
    """
    Build a diagonal Gaussian model per reference speaker.
    Equivalent to MATLAB's fit_reference_models().
    """
    models = []
    for path in reference_paths:
        signal = load_audio(path, sample_rate)

        # Build frame-level MFCC matrix (shape: [n_frames, n_mfcc])
        mfccs = librosa.feature.mfcc(
            y=signal.astype(float),
            sr=sample_rate,
            n_mfcc=N_MFCC + 1,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        ).T  # (n_frames, n_mfcc+1)
        mfccs = mfccs[:, 1:]  # drop energy coefficient

        # Spectral features per frame via embedding of individual frames
        # For speed we just use MFCC here (matches MATLAB's model.Mean/Var)
        mean = np.mean(mfccs, axis=0)
        var = np.var(mfccs, axis=0) + 1e-3

        models.append(SpeakerModel(name=path.stem, mean=mean, var=var))
        print(f"  Built model for: {path.stem}  ({len(mfccs)} frames)")

    return models


def score_segment_vs_model(
    segment_signal: np.ndarray,
    model: SpeakerModel,
    sample_rate: int,
) -> float:
    """
    Average diagonal Gaussian log-likelihood of segment frames vs model.
    Mirrors average_log_likelihood() in working.m.
    """
    mfccs = librosa.feature.mfcc(
        y=segment_signal.astype(float),
        sr=sample_rate,
        n_mfcc=N_MFCC + 1,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    ).T[:, 1:]  # (n_frames, n_mfcc)

    if mfccs.shape[0] == 0:
        return -1e9

    diff = mfccs - model.mean
    log_like = (
        -0.5 * np.sum(np.log(2.0 * np.pi * model.var))
        - 0.5 * np.sum(diff ** 2 / model.var, axis=1)
    )
    return float(np.mean(log_like))
