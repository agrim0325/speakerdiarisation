"""
Audio utilities: loading, resampling, normalization, feature extraction.
Mirrors the MATLAB helper functions from working.m.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import librosa
import soundfile as sf


SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".3gp", ".mp4"}


def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Load an audio file, convert to mono, resample to target SR, and normalize."""
    path = str(path)
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    return normalize_signal(audio)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Zero-mean and peak-normalize to ±0.95."""
    signal = signal - np.mean(signal)
    peak = np.max(np.abs(signal))
    if peak > 1e-9:
        return 0.95 * signal / peak
    return signal


def save_audio(path: str | Path, signal: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), signal.astype(np.float32), sample_rate)


def list_audio_files(directory: Path) -> list[Path]:
    """Return sorted list of audio files in a directory."""
    files = [
        f for f in sorted(directory.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.startswith(".")
    ]
    return files


def mix_microphones(mic1: np.ndarray, mic2: np.ndarray) -> np.ndarray:
    """Superposition-based mix: truncate to same length and average."""
    length = min(len(mic1), len(mic2))
    mixed = (mic1[:length] + mic2[:length]) / 2.0
    return normalize_signal(mixed)


def weighted_mix(chunk1: np.ndarray, chunk2: np.ndarray) -> np.ndarray:
    """RMS-weighted mix of two channel chunks for a single segment."""
    rms1 = np.sqrt(np.mean(chunk1 ** 2)) + 1e-9
    rms2 = np.sqrt(np.mean(chunk2 ** 2)) + 1e-9
    total = rms1 + rms2
    mixed = (rms1 / total) * chunk1 + (rms2 / total) * chunk2
    return normalize_signal(mixed)


def fade_edges(signal: np.ndarray, sample_rate: int, fade_seconds: float = 0.008) -> np.ndarray:
    """Apply a short linear fade-in and fade-out."""
    fade = min(int(fade_seconds * sample_rate), len(signal) // 4)
    if fade > 1:
        ramp = np.linspace(0, 1, fade)
        signal = signal.copy()
        signal[:fade] *= ramp
        signal[-fade:] *= ramp[::-1]
    return signal


def spectral_centroid_hz(signal: np.ndarray, sample_rate: int) -> float:
    """Compute spectral centroid in Hz using Hamming-windowed FFT."""
    if len(signal) == 0:
        return 0.0
    n_fft = max(1024, len(signal))
    window = np.hamming(len(signal))
    spectrum = np.abs(np.fft.rfft(signal * window, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    total = spectrum.sum()
    if total <= 1e-12:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


def estimate_delay_ms(
    chunk1: np.ndarray,
    chunk2: np.ndarray,
    sample_rate: int,
    max_delay_ms: float = 2.5,
) -> tuple[float, float]:
    """
    GCC-PHAT cross-correlation delay estimation.
    Returns (delay_ms, peak_correlation).
    """
    if len(chunk1) == 0 or len(chunk2) == 0:
        return 0.0, 0.0

    x = normalize_signal(chunk1)
    y = normalize_signal(chunk2)

    n_fft = 1
    while n_fft < len(x) + len(y):
        n_fft *= 2

    x_spec = np.fft.rfft(x, n=n_fft)
    y_spec = np.fft.rfft(y, n=n_fft)
    cross = x_spec * np.conj(y_spec)
    cross /= np.abs(cross) + 1e-9
    corr = np.fft.irfft(cross, n=n_fft).real

    max_shift = max(1, int(max_delay_ms / 1000.0 * sample_rate))
    corr_window = np.concatenate([corr[-max_shift:], corr[:max_shift + 1]])
    peak_idx = np.argmax(corr_window)
    peak_val = corr_window[peak_idx]
    best_shift = peak_idx - max_shift
    delay_ms = 1000.0 * best_shift / sample_rate
    return float(delay_ms), float(peak_val)
