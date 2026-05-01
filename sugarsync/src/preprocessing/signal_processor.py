"""
SugarSync — Signal Preprocessing
Detrending, bandpass filtering, peak detection, and SQI gating.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from typing import Tuple

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)
cfg = load_config()


# ── Mean Detrending ────────────────────────────────────────────────────────

def detrend_signal(signal: np.ndarray) -> np.ndarray:
    """
    Remove low-frequency baseline drift by subtracting the segment mean.
    This eliminates slow drift caused by differential tissue reflectance
    and thermal fluctuations.

    Parameters
    ----------
    signal : Raw ADC values (1D array)

    Returns
    -------
    Mean-detrended signal
    """
    return signal - np.mean(signal)


# ── Butterworth Bandpass Filter ────────────────────────────────────────────

def bandpass_filter(
    signal: np.ndarray,
    fs: float = 100.0,
    lowcut: float | None = None,
    highcut: float | None = None,
    order: int | None = None,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Default passband 0.5–5 Hz captures cardiac PPG frequencies (40–300 bpm)
    while rejecting motion artifacts and high-frequency noise.

    Parameters
    ----------
    signal  : Input signal (1D array)
    fs      : Sampling frequency in Hz
    lowcut  : Low cutoff Hz  (default from config)
    highcut : High cutoff Hz (default from config)
    order   : Filter order   (default from config)

    Returns
    -------
    Filtered signal (same shape as input)
    """
    filt_cfg  = cfg["preprocessing"]["filter"]
    lowcut  = lowcut  or filt_cfg["lowcut_hz"]
    highcut = highcut or filt_cfg["highcut_hz"]
    order   = order   or filt_cfg["order"]

    nyq = 0.5 * fs
    low = lowcut  / nyq
    high = highcut / nyq

    if not (0 < low < high < 1):
        raise ValueError(
            f"Invalid cutoff frequencies: low={lowcut}, high={highcut}, fs={fs}"
        )

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)       # zero-phase (forward-backward)


# ── Signal Quality Index ───────────────────────────────────────────────────

def compute_sqi(peaks: np.ndarray) -> float:
    """
    Compute Signal Quality Index from inter-beat interval consistency.

    SQI = max(0, 1 - σ(ΔRRI) / μ(ΔRRI))

    A perfect rhythm has SQI = 1.0. Segments below 0.6 are discarded.

    Parameters
    ----------
    peaks : Sample indices of detected PPG peaks

    Returns
    -------
    SQI in [0, 1]
    """
    if len(peaks) < 3:
        return 0.0

    ibi    = np.diff(peaks).astype(float)     # inter-beat intervals
    delta  = np.diff(ibi)                     # first-order differences
    mean_d = np.abs(np.mean(ibi))

    if mean_d < 1e-6:
        return 0.0

    sqi = max(0.0, 1.0 - np.std(delta) / mean_d)
    return float(sqi)


# ── Adaptive Peak Detection ────────────────────────────────────────────────

def detect_peaks(
    signal: np.ndarray,
    fs: float = 100.0,
) -> np.ndarray:
    """
    Adaptive PPG peak detection using scipy.find_peaks.

    - Height threshold = local mean × 1.08 (adaptive)
    - Min inter-peak distance = 40% of fs (40–200 bpm validity)
    - Spurious peaks removed by temporal regularity check

    Parameters
    ----------
    signal : Filtered PPG signal
    fs     : Sampling frequency

    Returns
    -------
    Array of peak sample indices
    """
    pd_cfg    = cfg["preprocessing"]["peak_detection"]
    min_dist  = int(pd_cfg["min_distance_factor"] * fs)

    # Adaptive height threshold
    height = np.mean(signal) + 0.08 * np.abs(np.mean(signal))

    peaks, properties = find_peaks(
        signal,
        height=height,
        distance=min_dist,
        prominence=0.01 * (np.max(signal) - np.min(signal)),
    )

    if len(peaks) < 2:
        return peaks

    # ── Spurious peak removal: IBI regularity check ──────────────────────
    ibi     = np.diff(peaks)
    median_ibi = np.median(ibi)
    valid   = [True]
    for i, interval in enumerate(ibi):
        # Accept peaks whose IBI is within ±50% of median
        if 0.5 * median_ibi <= interval <= 1.5 * median_ibi:
            valid.append(True)
        else:
            valid.append(False)

    peaks = peaks[np.array(valid, dtype=bool)]
    log.debug(f"Detected {len(peaks)} peaks after regularity filter.")
    return peaks


# ── Full Preprocessing Pipeline ────────────────────────────────────────────

def preprocess_segment(
    raw_signal: np.ndarray,
    fs: float = 100.0,
    sqi_threshold: float | None = None,
) -> Tuple[np.ndarray | None, np.ndarray, float]:
    """
    End-to-end preprocessing for a single signal segment.

    Steps:
        1. Mean detrending
        2. Butterworth bandpass filter (0.5–5 Hz, 4th order, zero-phase)
        3. Adaptive peak detection
        4. SQI computation and gating

    Parameters
    ----------
    raw_signal    : Raw ADC values from Arduino
    fs            : Sampling frequency (Hz)
    sqi_threshold : Minimum acceptable SQI (default from config)

    Returns
    -------
    filtered      : Filtered signal (or None if SQI too low)
    peaks         : Detected peak indices
    sqi           : Computed SQI value
    """
    sqi_threshold = sqi_threshold or cfg["sqi"]["min_threshold"]

    # Step 1: Detrend
    detrended = detrend_signal(raw_signal)

    # Step 2: Bandpass filter
    try:
        filtered = bandpass_filter(detrended, fs=fs)
    except Exception as exc:
        log.warning(f"Filter failed: {exc}. Returning None.")
        return None, np.array([]), 0.0

    # Step 3: Peak detection
    peaks = detect_peaks(filtered, fs=fs)

    # Step 4: SQI gate
    sqi = compute_sqi(peaks)
    if sqi < sqi_threshold:
        log.debug(f"Segment rejected: SQI={sqi:.3f} < {sqi_threshold}")
        return None, peaks, sqi

    return filtered, peaks, sqi
