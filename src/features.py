"""
Feature extraction for speaker identification.
Uses librosa for MFCCs, spectral rolloff, ZCR, and pitch.
Mirrors build_frame_feature_matrix() and compute_segment_features() from working.m.
"""

from __future__ import annotations

import numpy as np
import librosa

from .audio import normalize_signal, spectral_centroid_hz, estimate_delay_ms, weighted_mix


N_MFCC = 12
N_MELS = 24
N_FFT = 1024
HOP_LENGTH = int(0.010 * 16000)   # 10 ms hop
FRAME_LENGTH = int(0.030 * 16000) # 30 ms frame


def extract_speaker_embedding(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extract a fixed-length speaker embedding from an audio chunk.

    Returns a 1-D feature vector:
        [12 MFCC means, spectral centroid, spectral rolloff, ZCR, spectral flatness, pitch]
    = 17 features (matching MATLAB's featureMatrix columns)
    """
    signal = signal.ravel()
    if len(signal) < FRAME_LENGTH:
        signal = np.pad(signal, (0, FRAME_LENGTH - len(signal)))

    # MFCCs
    mfccs = librosa.feature.mfcc(
        y=signal.astype(float),
        sr=sample_rate,
        n_mfcc=N_MFCC + 1,       # +1 because we skip coeff 0 (energy)
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mfcc_mean = np.mean(mfccs[1:], axis=1)  # skip coefficient 0

    # Spectral features
    centroid = librosa.feature.spectral_centroid(
        y=signal.astype(float), sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=signal.astype(float), sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, roll_percent=0.85
    )
    zcr = librosa.feature.zero_crossing_rate(
        y=signal.astype(float), frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    flatness = librosa.feature.spectral_flatness(
        y=signal.astype(float), n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    # Pitch (fundamental frequency)
    f0, voiced_flag, _ = librosa.pyin(
        signal.astype(float),
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        fill_na=0.0,
    )
    pitch_mean = float(np.nanmean(f0[voiced_flag]) / 400.0) if voiced_flag.any() else 0.0

    return np.concatenate([
        mfcc_mean,
        [np.mean(centroid) / 4000.0],
        [np.mean(rolloff) / 8000.0],
        [np.mean(zcr)],
        [np.mean(flatness)],
        [pitch_mean],
    ])


def compute_segment_features(
    mic1: np.ndarray,
    mic2: np.ndarray,
    segments: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Compute per-segment feature matrix and audio chunks.

    Feature vector (18-D):
        [log_rms_ratio, log_rms_product, delay_ms, corr_peak,
         spectral_centroid_khz, duration_s,
         12 MFCC means]

    Returns:
        features: (N_segments, 18) array
        segment_audio: list of mixed mono arrays, one per segment
    """
    n = len(segments)
    features = np.zeros((n, 18))
    segment_audio: list[np.ndarray] = []

    for i, (t_start, t_end) in enumerate(segments):
        s = max(0, int(t_start * sample_rate))
        e = min(len(mic1), int(t_end * sample_rate))

        chunk1 = mic1[s:e]
        chunk2 = mic2[s:e]
        mixed = weighted_mix(chunk1, chunk2)
        segment_audio.append(mixed)

        rms1 = float(np.sqrt(np.mean(chunk1 ** 2))) + 1e-9
        rms2 = float(np.sqrt(np.mean(chunk2 ** 2))) + 1e-9
        delay_ms, corr_peak = estimate_delay_ms(chunk1, chunk2, sample_rate)

        # MFCC means for this segment (12 coefficients)
        embedding = extract_speaker_embedding(mixed, sample_rate)
        mfcc_part = embedding[:N_MFCC]

        features[i] = [
            np.log(rms1 / rms2),
            np.log(rms1 * rms2),
            delay_ms,
            corr_peak,
            spectral_centroid_hz(mixed, sample_rate) / 1000.0,
            t_end - t_start,
            *mfcc_part,
        ]

    return features, segment_audio
