<div align="center">

# 🩸 SugarSync
### Non-Invasive Glucose Monitoring via NIR-PPG & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![XGBoost MAE](https://img.shields.io/badge/XGBoost-MAE%209.12%20mg%2FdL-f97316?style=flat-square)](src/models/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.959-8b5cf6?style=flat-square)](src/models/)
[![IEEE](https://img.shields.io/badge/Format-IEEE-00629b?style=flat-square)](docs/)

> *Predict blood glucose levels non-invasively using near-infrared photoplethysmography signals and interpretable gradient-boosted machine learning — no needles, no strips, no pain.*

**Research(under review) · VIT Chennai · SCOPE Department**

</div>

---

## What is SugarSync?

SugarSync is a research prototype and open-source codebase for **non-invasive, continuous blood glucose estimation**. It uses a fingertip NIR-PPG sensor built around an **Arduino Uno + LM358 op-amp** to capture hemodynamic waveforms, extracts **40 physiologically-grounded features**, and feeds them into **XGBoost** and **LightGBM** models to predict glucose concentration in mg/dL — without a single finger prick.

This repository contains everything:
- Hardware specs and Arduino sketch
- Signal acquisition and real-time streaming code
- Full ML pipeline (preprocessing → features → training → inference)
- Explainability layer (SHAP, LIME, LRP)
- Synthetic data generator
- Local monitoring dashboard (Flask)
- Unit tests + GitHub Actions CI

---

## Key Results

| Model | Task | MAE | RMSE | R² | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | Regression | **9.12 mg/dL** | 11.45 mg/dL | **0.868** | — |
| LightGBM | Regression | 9.87 mg/dL | 12.21 mg/dL | — | — |
| LightGBM | Classification | — | — | — | **0.959** |

Classification accuracy exceeded **86%** across Low / Normal / High glucose categories.

---
## How It Works

**1 → Hardware**
Fingertip clip sensor with Red (660nm) + NIR (940nm) LEDs reads blood volume changes. Signal is amplified by an LM358 op-amp and digitised by the Arduino Uno's 10-bit ADC at 115200 baud.

**2 → Python Signal Pipeline**
- Mean detrending — removes baseline drift from tissue reflectance changes
- Butterworth bandpass filter — 0.5 to 5 Hz, 4th order, zero-phase (filtfilt)
- Adaptive peak detection — scipy.signal.find_peaks with local amplitude threshold
- Signal Quality Index gate — segments with SQI below 0.6 are discarded

**3 → Feature Extraction** — 40 features from 3-second overlapping windows
- Time-domain: mean, std, variance, skewness, kurtosis, min, max, range, Q25, Q50, Q75, IQR
- Morphological: AC/DC ratio, pulse width, systolic peak, diastolic valley, ACPP, rise time, fall time
- Frequency-domain: LF power, HF power, LF/HF ratio, total power, spectral centroid, entropy
- HRV: HR (bpm), RMSSD, SDNN, PPI mean, PPI std, pNN50, SQI, beat count

**4 → ML Models**
- XGBoost Regressor → predicts continuous glucose value in mg/dL
- LightGBM Classifier → predicts category (Low / Normal / High)

**5 → XAI Layer**
- SHAP — global and local feature importance via Shapley values
- LIME — instance-level surrogate linear model explanations
- LRP — layer-wise relevance propagation on a shallow neural network

---

## Repository Structure

```
sugarsync/
├── arduino/
│   └── ppg_sensor.ino
├── src/
│   ├── acquisition/
│   │   └── serial_reader.py
│   ├── preprocessing/
│   │   └── signal_processor.py
│   ├── features/
│   │   └── feature_pipeline.py
│   ├── models/
│   │   ├── train_xgboost.py
│   │   ├── train_lightgbm.py
│   │   └── predict.py
│   ├── explainability/
│   │   ├── shap_analysis.py
│   │   └── lime_analysis.py
│   └── utils/
│       ├── config.py
│       └── logger.py
├── dashboard/
│   ├── app.py
│   └── templates/index.html
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── tests/
│   └── test_pipeline.py
├── config.yaml
├── requirements.txt
└── environment.yml
```

## Quick Start

### 1. Clone

```bash
git clone https://github.com/isha271/sugarsync.git
cd sugarsync
```

### 2. Install

```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate sugarsync
```

### 3. Run the Dashboard (no hardware needed)

```bash
python dashboard/app.py
```

Open **http://localhost:5000** — runs in demo mode with simulated glucose data.

## Hardware Bill of Materials

| Component | Spec | Approx. Cost |
|---|---|---|
| Arduino Uno (ATmega328P) | 16 MHz, 10-bit ADC | ₹450 |
| NIR-PPG Clip Sensor | Red 660nm + NIR 940nm LEDs + photodiode | ₹300 |
| LM358 Dual Op-Amp | Transimpedance + gain stage (AFE) | ₹15 |
| 16×2 LCD Module | I2C, real-time HR display | ₹120 |
| Piezoelectric Buzzer | Optional beat feedback | ₹20 |
| Resistors, caps, breadboard | — | ₹80 |
| **Total** | | **~₹985** |

---

## Feature Reference (40 total)

| Domain | Count | Features |
|---|---|---|
| Time Domain | 12 | mean, std, variance, skewness, kurtosis, min, max, range, Q25, Q50, Q75, IQR |
| Morphological | 10 | AC component, DC component, AC/DC ratio, ACPP, systolic peak, diastolic valley, pulse width (50%), rise time, fall time, valley depth |
| Frequency Domain | 8 | LF power, HF power, LF/HF ratio, total power, spectral centroid, spectral bandwidth, dominant frequency, spectral entropy |
| HRV & Quality | 10 | HR (bpm), RMSSD, SDNN, PPI mean, PPI std, pNN50, SQI, beat count, IBI regularity, peak confidence |

---

## Model Details

### XGBoost Regressor
```python
XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1,   # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    early_stopping_rounds=50, eval_metric='mae'
)
```

### LightGBM Classifier
```python
LGBMClassifier(
    n_estimators=500, num_leaves=63, learning_rate=0.05,
    feature_fraction=0.8, bagging_fraction=0.8,
    class_weight='balanced', early_stopping_rounds=50
)
```

**Data split:** 70% train / 15% validation / 15% test — no subject overlap across partitions.

---

## Explainability

SugarSync is built to be clinically interpretable, not just accurate.

**SHAP** — Top global predictors: `Age`, `Mean`, `Q25`, `AC_Component`, `HRV_RMSSD`. Higher HRV → lower predicted glucose (parasympathetic activation in normoglycaemia). Higher mean amplitude + lower variability → hyperglycaemic prediction (vascular stiffening pattern).

**LIME** — Instance-level: low HRV + widened pulse width → positive glucose contribution. Consistent with known haemodynamic mechanisms in diabetes.

**LRP** — Highest relevance features: `Signal_Quality_Index`, `Pulse_Width`, `Diastolic_Valley`, `AC/DC ratio`. Waveform amplitude and systolic-diastolic asymmetry carry the most glucose-linked haemodynamic signal.

---

## Limitations & Future Work

- Dataset collected in a controlled lab environment — real-world generalisation requires larger, multi-site clinical trials
- Validation against ISO 15197 / Clarke Error Grid / Bland-Altman analysis is pending
- Calibration drift across sessions not yet addressed
- Planned: temperature sensor fusion, wrist-form wearable design, anemia detection via nail-bed CNN module
