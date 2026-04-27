"""
Speech segmentation using short-time energy and Hamming windowing.
Mirrors detect_segments() from working.m.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def detect_segments(
    signal: np.ndarray,
    sample_rate: int,
    energy_threshold_pct: float = 1.5,
    min_segment_duration: float = 0.20,
    merge_gap: float = 0.50,
) -> np.ndarray:
    """
    Detect speech segments via short-time energy with Hamming windowing.

    Returns an (N, 2) array of [start_sec, end_sec] pairs.
    """
    frame_size = int(0.025 * sample_rate)
    hop_size = int(0.010 * sample_rate)

    energy = _short_time_energy(signal, frame_size, hop_size)
    if len(energy) == 0:
        return np.zeros((0, 2))

    # Median filter to smooth energy (kernel=30, matching MATLAB fix)
    energy = median_filter(energy, size=30)

    e_max = energy.max()
    if e_max <= 1e-12:
        return np.zeros((0, 2))

    threshold = (energy_threshold_pct / 100.0) * e_max
    is_speech = energy > threshold

    # Collect raw segments
    starts, ends = [], []
    in_seg = False
    start_frame = 0
    for i, active in enumerate(is_speech):
        if active and not in_seg:
            in_seg = True
            start_frame = i
        elif not active and in_seg:
            in_seg = False
            t_start = start_frame * hop_size / sample_rate
            t_end = i * hop_size / sample_rate
            if (t_end - t_start) >= min_segment_duration:
                starts.append(t_start)
                ends.append(t_end)
    if in_seg:
        t_start = start_frame * hop_size / sample_rate
        t_end = len(is_speech) * hop_size / sample_rate
        if (t_end - t_start) >= min_segment_duration:
            starts.append(t_start)
            ends.append(t_end)

    if not starts:
        return np.zeros((0, 2))

    # Merge close segments
    merged = [[starts[0], ends[0]]]
    for s, e in zip(starts[1:], ends[1:]):
        if (s - merged[-1][1]) < merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    return np.array(merged, dtype=float)


def _short_time_energy(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """Frame-level short-time energy using a Hamming window."""
    signal = signal.ravel()
    if len(signal) == 0:
        return np.array([])

    window = np.hamming(frame_size)

    if len(signal) < frame_size:
        padded = np.zeros(frame_size)
        padded[:len(signal)] = signal
        return np.array([np.sum((padded * window) ** 2)])

    num_frames = 1 + (len(signal) - frame_size) // hop_size
    energy = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start:start + frame_size]
        energy[i] = np.sum((frame * window) ** 2)

    return energy
