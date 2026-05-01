"""
SugarSync — Unit Tests
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.preprocessing.signal_processor import (
    detrend_signal,
    bandpass_filter,
    compute_sqi,
    detect_peaks,
    preprocess_segment,
)
from src.features.feature_pipeline import (
    extract_time_domain,
    extract_morphological,
    extract_frequency_domain,
    extract_hrv_quality,
    extract_features,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_ppg():
    """Generate a clean synthetic 6-second PPG signal at 100 Hz."""
    fs      = 100.0
    t       = np.linspace(0, 6, int(6 * fs))
    hr_hz   = 1.2   # 72 bpm
    ppg     = (
        0.8 * np.sin(2 * np.pi * hr_hz * t)
        + 0.3 * np.sin(2 * np.pi * 2 * hr_hz * t)
        + 0.05 * np.random.randn(len(t))
        + 0.1 * t   # slight drift
    )
    return ppg.astype(np.float32)


@pytest.fixture
def clean_peaks(synthetic_ppg):
    _, peaks, _ = preprocess_segment(synthetic_ppg)
    return peaks


# ── Preprocessing Tests ────────────────────────────────────────────────────

class TestDetrend:
    def test_mean_approximately_zero(self, synthetic_ppg):
        out = detrend_signal(synthetic_ppg)
        assert abs(np.mean(out)) < 1e-6

    def test_preserves_shape(self, synthetic_ppg):
        out = detrend_signal(synthetic_ppg)
        assert out.shape == synthetic_ppg.shape


class TestBandpass:
    def test_output_shape_preserved(self, synthetic_ppg):
        out = bandpass_filter(synthetic_ppg)
        assert out.shape == synthetic_ppg.shape

    def test_high_freq_attenuated(self):
        fs   = 100.0
        t    = np.linspace(0, 6, 600)
        hf   = np.sin(2 * np.pi * 40 * t)   # 40 Hz — should be attenuated
        out  = bandpass_filter(hf, fs=fs)
        assert np.std(out) < 0.05 * np.std(hf), "High-freq component not attenuated"

    def test_invalid_cutoffs_raise(self):
        with pytest.raises(ValueError):
            bandpass_filter(np.zeros(600), lowcut=10.0, highcut=1.0)


class TestSQI:
    def test_perfect_rhythm_sqi_near_one(self):
        peaks = np.array([100, 200, 300, 400, 500])  # perfect 100-sample intervals
        sqi   = compute_sqi(peaks)
        assert sqi > 0.9

    def test_irregular_rhythm_low_sqi(self):
        peaks = np.array([100, 130, 300, 450, 600])  # irregular
        sqi   = compute_sqi(peaks)
        assert sqi < 0.7

    def test_too_few_peaks_returns_zero(self):
        assert compute_sqi(np.array([50])) == 0.0

    def test_sqi_in_range(self, clean_peaks):
        sqi = compute_sqi(clean_peaks)
        assert 0.0 <= sqi <= 1.0


class TestPeakDetection:
    def test_detects_peaks(self, synthetic_ppg):
        _, peaks, _ = preprocess_segment(synthetic_ppg)
        # Expect ~7 peaks for 72 bpm over 6 seconds
        assert 4 <= len(peaks) <= 12

    def test_hr_within_physiological_range(self, synthetic_ppg):
        fs = 100.0
        _, peaks, _ = preprocess_segment(synthetic_ppg)
        if len(peaks) >= 2:
            ibi = np.diff(peaks) / fs
            hr  = 60.0 / ibi
            assert np.all(hr >= 40) and np.all(hr <= 200)


class TestPreprocessPipeline:
    def test_valid_segment_returns_tuple(self, synthetic_ppg):
        filtered, peaks, sqi = preprocess_segment(synthetic_ppg)
        assert filtered is not None
        assert len(peaks) > 0
        assert 0.0 <= sqi <= 1.0

    def test_noise_only_rejected(self):
        noise = np.random.randn(600) * 0.001   # flat noise — no peaks
        filtered, peaks, sqi = preprocess_segment(noise, sqi_threshold=0.60)
        # Should fail SQI gate
        assert filtered is None or sqi < 0.60


# ── Feature Extraction Tests ───────────────────────────────────────────────

class TestTimeDomain:
    def test_returns_12_features(self, synthetic_ppg):
        feats = extract_time_domain(synthetic_ppg)
        assert len(feats) == 12

    def test_mean_correct(self, synthetic_ppg):
        feats = extract_time_domain(synthetic_ppg)
        assert abs(feats["mean"] - float(np.mean(synthetic_ppg))) < 1e-4

    def test_iqr_positive(self, synthetic_ppg):
        feats = extract_time_domain(synthetic_ppg)
        assert feats["iqr"] > 0


class TestMorphological:
    def test_returns_10_features(self, synthetic_ppg, clean_peaks):
        feats = extract_morphological(synthetic_ppg, clean_peaks)
        assert len(feats) == 10

    def test_ac_dc_ratio_positive(self, synthetic_ppg, clean_peaks):
        feats = extract_morphological(synthetic_ppg, clean_peaks)
        assert feats["ac_dc_ratio"] >= 0

    def test_pulse_height_nonnegative(self, synthetic_ppg, clean_peaks):
        feats = extract_morphological(synthetic_ppg, clean_peaks)
        assert feats["pulse_height_acpp"] >= 0


class TestFrequencyDomain:
    def test_returns_8_features(self, synthetic_ppg):
        feats = extract_frequency_domain(synthetic_ppg)
        assert len(feats) == 8

    def test_lf_hf_ratio_positive(self, synthetic_ppg):
        feats = extract_frequency_domain(synthetic_ppg)
        assert feats["lf_hf_ratio"] >= 0

    def test_spectral_entropy_positive(self, synthetic_ppg):
        feats = extract_frequency_domain(synthetic_ppg)
        assert feats["spectral_entropy"] > 0


class TestHRVQuality:
    def test_returns_10_features(self, clean_peaks):
        feats = extract_hrv_quality(clean_peaks, sqi=0.85)
        assert len(feats) == 10

    def test_hr_within_range(self, clean_peaks):
        feats = extract_hrv_quality(clean_peaks, sqi=0.85)
        if feats["hr_bpm"] > 0:
            assert 40 <= feats["hr_bpm"] <= 200

    def test_no_peaks_returns_zeros(self):
        feats = extract_hrv_quality(np.array([]), sqi=0.0)
        assert all(v == 0.0 for v in feats.values())


class TestFullPipeline:
    def test_extract_features_returns_dict(self, synthetic_ppg):
        feats = extract_features(synthetic_ppg)
        assert feats is not None
        assert isinstance(feats, dict)

    def test_at_least_40_features(self, synthetic_ppg):
        feats = extract_features(synthetic_ppg)
        if feats:
            assert len(feats) >= 40

    def test_context_features_included(self, synthetic_ppg):
        ctx   = {"age": 35, "weight": 70.0, "diabetic": 1}
        feats = extract_features(synthetic_ppg, context=ctx)
        if feats:
            assert "age" in feats
            assert feats["age"] == 35.0
