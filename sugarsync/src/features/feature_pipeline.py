"""
SugarSync — Feature Extraction Pipeline
Extracts 40 physiologically-grounded features from preprocessed PPG segments.

Feature domains:
    - Time domain (12):      statistical moments and order statistics
    - Morphological (10):    waveform shape and vascular indices
    - Frequency domain (8):  LF/HF power, spectral descriptors
    - HRV & Quality (10):    heart rate variability, signal quality metrics

Reference: Section VI (Methodology) of the SugarSync paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from typing import Dict, List

from src.preprocessing.signal_processor import preprocess_segment, detect_peaks
from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)
cfg = load_config()


# ─────────────────────────────────────────────────────────────────────────────
# Time-Domain Features
# ─────────────────────────────────────────────────────────────────────────────

def extract_time_domain(signal: np.ndarray) -> Dict[str, float]:
    """
    Statistical moments and order statistics over the signal window.

    Features
    --------
    mean, std, variance, skewness, kurtosis,
    min, max, range, q25, q50 (median), q75, iqr
    """
    q25, q50, q75 = np.percentile(signal, [25, 50, 75])
    return {
        "mean":     float(np.mean(signal)),
        "std":      float(np.std(signal)),
        "variance": float(np.var(signal)),
        "skewness": float(skew(signal)),
        "kurtosis": float(kurtosis(signal)),
        "min":      float(np.min(signal)),
        "max":      float(np.max(signal)),
        "range":    float(np.ptp(signal)),
        "q25":      float(q25),
        "q50":      float(q50),
        "q75":      float(q75),
        "iqr":      float(q75 - q25),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Morphological Features
# ─────────────────────────────────────────────────────────────────────────────

def extract_morphological(
    signal: np.ndarray,
    peaks: np.ndarray,
    fs: float = 100.0,
) -> Dict[str, float]:
    """
    Waveform-shape features capturing vascular dynamics.

    The AC component (pulsatile amplitude) and DC component (baseline)
    reflect hemodynamic and optical properties linked to blood glucose.

    Features
    --------
    ac_component, dc_component, ac_dc_ratio,
    pulse_height_acpp, systolic_peak, diastolic_valley,
    pulse_width_50pct, rise_time, fall_time, valley_depth
    """
    dc_comp = float(np.mean(np.abs(signal)))
    ac_comp = float(np.std(signal))             # RMS of AC component

    ac_dc_ratio    = ac_comp / (dc_comp + 1e-8)
    systolic_peak  = float(np.max(signal))
    diastolic_val  = float(np.min(signal))
    pulse_height   = systolic_peak - diastolic_val   # ACPP

    # Pulse width at 50% amplitude (across all detected beats)
    pulse_width_50 = 0.0
    rise_times     = []
    fall_times     = []

    if len(peaks) >= 2:
        widths = []
        for i, pk in enumerate(peaks[:-1]):
            next_pk = peaks[i + 1]
            beat    = signal[pk:next_pk]
            if len(beat) < 4:
                continue

            half_amp = (np.max(beat) + np.min(beat)) / 2
            above    = np.where(beat >= half_amp)[0]
            if len(above) >= 2:
                widths.append((above[-1] - above[0]) / fs * 1000)   # ms

            # Rise and fall time within the beat
            pk_local = int(np.argmax(beat))
            rise_times.append(pk_local / fs * 1000)
            fall_times.append((len(beat) - pk_local) / fs * 1000)

        pulse_width_50 = float(np.mean(widths)) if widths else 0.0

    return {
        "ac_component":      ac_comp,
        "dc_component":      dc_comp,
        "ac_dc_ratio":       ac_dc_ratio,
        "pulse_height_acpp": pulse_height,
        "systolic_peak":     systolic_peak,
        "diastolic_valley":  diastolic_val,
        "pulse_width_50pct": pulse_width_50,
        "rise_time":         float(np.mean(rise_times))  if rise_times  else 0.0,
        "fall_time":         float(np.mean(fall_times))  if fall_times  else 0.0,
        "valley_depth":      float(np.abs(diastolic_val)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-Domain Features
# ─────────────────────────────────────────────────────────────────────────────

def extract_frequency_domain(
    signal: np.ndarray,
    fs: float = 100.0,
) -> Dict[str, float]:
    """
    Spectral features via Welch's power spectral density estimate.

    LF band (0.04–0.15 Hz) reflects sympathetic modulation.
    HF band (0.15–0.40 Hz) reflects parasympathetic (vagal) tone.
    LF/HF ratio is a standard autonomic balance metric.

    Features
    --------
    lf_power, hf_power, lf_hf_ratio, total_power,
    spectral_centroid, spectral_bandwidth, dominant_frequency, spectral_entropy
    """
    freq_cfg = cfg["features"]["frequency_domain"]
    lf_low, lf_high = freq_cfg["lf_band"]
    hf_low, hf_high = freq_cfg["hf_band"]

    nperseg = min(len(signal), 256)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    def band_power(f_low: float, f_high: float) -> float:
        mask = (freqs >= f_low) & (freqs <= f_high)
        return float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0

    lf_power    = band_power(lf_low, lf_high)
    hf_power    = band_power(hf_low, hf_high)
    total_power = float(np.trapz(psd, freqs))
    lf_hf_ratio = lf_power / (hf_power + 1e-8)

    # Spectral centroid
    centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-8))

    # Spectral bandwidth (weighted standard deviation around centroid)
    bandwidth = float(
        np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / (np.sum(psd) + 1e-8))
    )

    # Dominant frequency
    dom_freq = float(freqs[np.argmax(psd)])

    # Spectral entropy (normalized)
    psd_norm = psd / (np.sum(psd) + 1e-8)
    spec_ent = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-8)))

    return {
        "lf_power":             lf_power,
        "hf_power":             hf_power,
        "lf_hf_ratio":          lf_hf_ratio,
        "total_power":          total_power,
        "spectral_centroid":    centroid,
        "spectral_bandwidth":   bandwidth,
        "dominant_frequency":   dom_freq,
        "spectral_entropy":     spec_ent,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HRV & Quality Features
# ─────────────────────────────────────────────────────────────────────────────

def extract_hrv_quality(
    peaks: np.ndarray,
    sqi: float,
    fs: float = 100.0,
) -> Dict[str, float]:
    """
    Heart Rate Variability metrics and signal quality metadata.

    HR = 60 / RR̄  (from PPI peaks)
    RMSSD = √(1/(N-1) Σ(RRᵢ₊₁ − RRᵢ)²)  — parasympathetic index

    Features
    --------
    hr_bpm, rmssd, sdnn, ppi_mean, ppi_std,
    pnn50, sqi, beat_count, ibi_regularity, peak_confidence
    """
    if len(peaks) < 2:
        return {k: 0.0 for k in [
            "hr_bpm", "rmssd", "sdnn", "ppi_mean", "ppi_std",
            "pnn50", "sqi", "beat_count", "ibi_regularity", "peak_confidence"
        ]}

    ppi    = np.diff(peaks) / fs * 1000       # inter-beat intervals in ms
    nn50   = np.sum(np.abs(np.diff(ppi)) > 50)
    pnn50  = nn50 / (len(ppi) - 1 + 1e-8) * 100

    rr_mean = np.mean(ppi)
    hr_bpm  = 60_000 / (rr_mean + 1e-8)

    # RMSSD
    rmssd = float(
        np.sqrt(np.mean(np.diff(ppi) ** 2))
    ) if len(ppi) > 1 else 0.0

    # IBI regularity: inverse of coefficient of variation
    cv = np.std(ppi) / (rr_mean + 1e-8)
    regularity = float(max(0.0, 1.0 - cv))

    # Peak confidence: mean height relative to signal noise
    peak_conf = float(np.clip(sqi, 0, 1))

    return {
        "hr_bpm":          float(np.clip(hr_bpm, 0, 250)),
        "rmssd":           rmssd,
        "sdnn":            float(np.std(ppi)),
        "ppi_mean":        float(rr_mean),
        "ppi_std":         float(np.std(ppi)),
        "pnn50":           float(pnn50),
        "sqi":             float(sqi),
        "beat_count":      float(len(peaks)),
        "ibi_regularity":  regularity,
        "peak_confidence": peak_conf,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Master Feature Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    raw_segment: np.ndarray,
    fs: float = 100.0,
    context: Dict | None = None,
) -> Dict[str, float] | None:
    """
    Full pipeline: preprocess → extract all 40 features.

    Parameters
    ----------
    raw_segment : Raw ADC values from a 6-second window
    fs          : Sampling frequency (Hz)
    context     : Optional dict with behavioral data:
                  {age, weight, height, diabetic, meal_gap_min,
                   sleep_duration_h, transpiration, bmi}

    Returns
    -------
    dict of 40 features, or None if the segment fails SQI gate
    """
    filtered, peaks, sqi = preprocess_segment(raw_segment, fs=fs)

    if filtered is None:
        log.debug("Segment rejected at SQI gate.")
        return None

    # 3-second overlapping sub-windows for feature averaging
    sub_len   = int(fs * cfg["features"]["subwindow_size_s"])
    step      = int(sub_len * (1 - cfg["features"]["overlap_fraction"]))

    sub_features: List[Dict] = []
    for start in range(0, len(filtered) - sub_len + 1, step):
        seg  = filtered[start : start + sub_len]
        seg_peaks = peaks[(peaks >= start) & (peaks < start + sub_len)] - start

        td   = extract_time_domain(seg)
        morph = extract_morphological(seg, seg_peaks, fs)
        freq  = extract_frequency_domain(seg, fs)
        hrv   = extract_hrv_quality(seg_peaks, sqi, fs)

        sub_features.append({**td, **morph, **freq, **hrv})

    if not sub_features:
        return None

    # Average across sub-windows
    feature_df = pd.DataFrame(sub_features)
    features   = feature_df.mean().to_dict()

    # Attach context features
    if context:
        for key in ["age", "weight", "height", "diabetic",
                    "meal_gap_min", "sleep_duration_h", "bmi", "transpiration"]:
            features[key] = float(context.get(key, 0.0))

    log.debug(f"Extracted {len(features)} features from segment.")
    return features


def batch_extract(
    raw_df: pd.DataFrame,
    fs: float = 100.0,
    context: Dict | None = None,
    window_col: str = "nir_adc",
    label_col: str | None = "glucometer_mg_dl",
) -> pd.DataFrame:
    """
    Extract features from all 6-second windows in a raw DataFrame.

    Parameters
    ----------
    raw_df     : DataFrame with timestamp_ms, nir_adc, red_adc columns
    fs         : Sampling frequency
    context    : Behavioral context dict (applied to all windows)
    window_col : Which ADC channel to use for feature extraction
    label_col  : Ground truth column name (included if present)

    Returns
    -------
    DataFrame where each row is one valid feature vector
    """
    window_size = int(fs * cfg["features"]["window_size_s"])
    records: List[Dict] = []

    values = raw_df[window_col].values

    for i in range(0, len(values) - window_size + 1, window_size // 2):
        window = values[i : i + window_size]
        feats  = extract_features(window, fs=fs, context=context)

        if feats is None:
            continue

        if label_col and label_col in raw_df.columns:
            # Use the glucose value at the midpoint of this window
            mid_idx = i + window_size // 2
            feats["glucose_mg_dl"] = float(raw_df.iloc[mid_idx][label_col])

        records.append(feats)

    log.info(f"Extracted {len(records)} valid feature vectors from {len(values)} samples.")
    return pd.DataFrame(records)
