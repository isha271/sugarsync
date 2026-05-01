<div align="center">

<img src="assets/banner.svg" alt="SugarSync Banner" width="100%"/>

# SugarSync
### Non-Invasive Glucose Monitoring via NIR-PPG & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![IEEE](https://img.shields.io/badge/Format-IEEE-00629b?style=flat-square)](docs/paper/)
[![XGBoost](https://img.shields.io/badge/XGBoost-MAE%209.12-f97316?style=flat-square)](src/models/)
[![AUC](https://img.shields.io/badge/ROC--AUC-0.959-8b5cf6?style=flat-square)](src/models/)

> *Predict blood glucose levels non-invasively using near-infrared photoplethysmography signals and interpretable gradient-boosted machine learning — no needles, no strips, no pain.*

</div>

---

## What is SugarSync?

SugarSync is a research prototype and open-source codebase for **non-invasive, continuous blood glucose estimation**. It uses a fingertip NIR-PPG sensor (built around an Arduino Uno + LM358 op-amp) to capture hemodynamic waveforms, extracts 40 physiologically-grounded features, and feeds them into XGBoost and LightGBM models to predict glucose concentration in mg/dL — without a single finger prick.

This repository contains everything: hardware specs, signal acquisition code, full ML pipeline, explainability layer (SHAP, LIME, LRP), a synthetic data generator, and a local monitoring dashboard.

**Published as an IEEE conference paper** by Dr. S.A. Sajidha, Shriyansh Patnaik, Ananya Tripathi, and Isha Shrivastava — SCOPE, Vellore Institute of Technology, Chennai.

---

## Key Results

| Model | Task | MAE (mg/dL) | RMSE (mg/dL) | R² | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | Regression | **9.12** | 11.45 | **0.868** | — |
| LightGBM | Regression | 9.87 | 12.21 | — | — |
| LightGBM | Classification | — | — | — | **0.959** |

Classification accuracy exceeded **86%** across low/normal/high glucose categories.

---

## How It Works

```
Fingertip PPG Sensor (NIR 940nm + Red 660nm)
        │
        ▼
LM358 Op-Amp → Arduino Uno ADC (10-bit, 115200 baud)
        │
        ▼
Python Signal Pipeline
  ├── Mean detrending
  ├── Butterworth bandpass filter (0.5–5 Hz, 4th order, zero-phase)
  ├── Adaptive peak detection (scipy.signal.find_peaks)
  └── Signal Quality Index (SQI ≥ 0.6 gate)
        │
        ▼
Feature Extraction (40 features, 3-second overlapping windows)
  ├── Time-domain: mean, variance, skewness, kurtosis, Q25, Q75, range
  ├── Morphological: AC/DC ratio, pulse width, systolic peak, diastolic valley, ACPP
  ├── Frequency-domain: LF power, HF power, LF/HF ratio, spectral centroid
  └── HRV: HR (bpm), RMSSD, PPI intervals
        │
        ▼
ML Models
  ├── XGBoost Regressor → glucose value (mg/dL)
  └── LightGBM Classifier → category (Low / Normal / High)
        │
        ▼
XAI Layer
  ├── SHAP (global + local feature importance)
  ├── LIME (instance-level surrogate explanations)
  └── LRP (layer-wise relevance propagation on shallow NN)
```

---

## Repository Structure

```
sugarsync/
├── src/
│   ├── acquisition/          # Arduino serial reader & real-time streamer
│   │   ├── serial_reader.py
│   │   └── live_visualizer.py
│   ├── preprocessing/        # Signal cleaning & quality gating
│   │   ├── detrend.py
│   │   ├── bandpass_filter.py
│   │   ├── peak_detector.py
│   │   └── sqi.py
│   ├── features/             # Feature extraction pipeline
│   │   ├── time_domain.py
│   │   ├── morphological.py
│   │   ├── frequency_domain.py
│   │   └── feature_pipeline.py
│   ├── models/               # Model training, evaluation, inference
│   │   ├── train_xgboost.py
│   │   ├── train_lightgbm.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   ├── explainability/       # SHAP, LIME, LRP
│   │   ├── shap_analysis.py
│   │   ├── lime_analysis.py
│   │   └── lrp_analysis.py
│   └── utils/                # Shared helpers
│       ├── config.py
│       ├── logger.py
│       └── data_loader.py
├── data/
│   ├── raw/                  # Place your .xlsx signal exports here
│   ├── processed/            # Auto-generated cleaned datasets
│   └── synthetic/            # Augmented training data
├── dashboard/                # Local Flask/HTML monitoring dashboard
│   ├── app.py
│   ├── templates/
│   └── static/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_explainability.ipynb
├── tests/                    # Unit tests
├── arduino/                  # Arduino .ino sketch
│   └── ppg_sensor.ino
├── assets/                   # Banner, figures
├── docs/                     # Paper PDF, hardware schematic
├── requirements.txt
├── environment.yml
├── config.yaml
└── README.md
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/isha271/sugarsync.git
cd sugarsync

# Using conda (recommended)
conda env create -f environment.yml
conda activate sugarsync

# Or pip
pip install -r requirements.txt
```

### 2. Hardware Setup (Optional — skip for demo mode)

Flash the Arduino sketch:
```
arduino/ppg_sensor.ino → Upload to Arduino Uno
Connect: NIR-PPG sensor → LM358 AFE → Arduino analog pin A0
```

Find your COM port:
```bash
python src/acquisition/serial_reader.py --list-ports
```

### 3. Collect Data (Live)

```bash
python src/acquisition/serial_reader.py \
    --port COM3 \
    --baud 115200 \
    --output data/raw/session_001.xlsx \
    --duration 120
```

### 4. Run the Full Pipeline

```bash
# Preprocess + extract features
python src/features/feature_pipeline.py \
    --input data/raw/session_001.xlsx \
    --output data/processed/features_001.csv

# Train models
python src/models/train_xgboost.py --data data/processed/features_001.csv
python src/models/train_lightgbm.py --data data/processed/features_001.csv

# Predict on new data
python src/models/predict.py --input data/processed/features_new.csv
```

### 5. Explainability Reports

```bash
python src/explainability/shap_analysis.py --model models/xgboost_regressor.pkl
python src/explainability/lime_analysis.py --model models/lightgbm_classifier.pkl --index 42
```

### 6. Launch Dashboard

```bash
python dashboard/app.py
# Open http://localhost:5000
```

---

## Hardware Bill of Materials

| Component | Spec | Approx. Cost |
|---|---|---|
| Arduino Uno (ATmega328P) | 16 MHz, 10-bit ADC | ₹450 |
| NIR-PPG Clip Sensor | Red 660nm + NIR 940nm | ₹300 |
| LM358 Dual Op-Amp | Transimpedance + gain stage | ₹15 |
| 16x2 LCD Module | I2C interface | ₹120 |
| Piezoelectric Buzzer | Feedback | ₹20 |
| Resistors, caps, breadboard | — | ₹80 |
| **Total** | | **~₹985** |

Full schematic: [`docs/hardware_schematic.pdf`](docs/)

---

## Feature Reference

All 40 features extracted per 3-second window:

**Time Domain (12):** mean, std, variance, skewness, kurtosis, min, max, range, Q25, Q50, Q75, IQR

**Morphological (10):** AC component, DC component, AC/DC ratio, pulse height (ACPP), systolic peak, diastolic valley, pulse width (50%), rise time, fall time, valley depth

**Frequency Domain (8):** LF power (0.04–0.15 Hz), HF power (0.15–0.4 Hz), LF/HF ratio, total power, spectral centroid, spectral bandwidth, dominant frequency, spectral entropy

**HRV & Quality (10):** HR (bpm), RMSSD, SDNN, PPI mean, PPI std, pNN50, SQI, beat count, inter-beat regularity, peak confidence

---

## Model Details

### XGBoost Regressor

```python
XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,      # L1
    reg_lambda=1.0,     # L2
    early_stopping_rounds=50,
    eval_metric='mae'
)
```

### LightGBM Classifier

```python
LGBMClassifier(
    n_estimators=500,
    num_leaves=63,
    learning_rate=0.05,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    class_weight='balanced',
    verbose=-1
)
```

Data split: **70% train / 15% validation / 15% test** — no subject overlap across partitions.

---

## Explainability

SugarSync prioritizes clinical interpretability alongside predictive performance.

**SHAP** — Top global predictors: `Age`, `Mean`, `Q25`, `AC_Component`, `HRV_RMSSD`. Higher HRV → lower predicted glucose (parasympathetic activation in normoglycaemia). Higher mean amplitude + lower variability → hyperglycaemic prediction (vascular stiffening).

**LIME** — Instance-level: low HRV + widened pulse width → positive glucose contribution. Consistent with known haemodynamic mechanisms.

**LRP** — Highest relevance: `Signal_Quality_Index`, `Pulse_Width`, `Diastolic_Valley`, `AC/DC ratio`. Waveform amplitude and systolic-diastolic asymmetry encode the most haemodynamic glucose signal.

---

## Data Augmentation

Raw dataset: **1,100 samples** → Augmented to **2,484 samples** using controlled bootstrapping with GPT-based generative augmentation. Physiological bounds enforced:

- Glucose: 70–300 mg/dL
- SQI: 0.6–0.98  
- Meal gap: 15–360 min

Distribution comparison plot: [`assets/synthetic_vs_original.png`](assets/)

---

## Limitations & Future Work

- Dataset collected in a controlled lab environment; real-world generalization requires larger, multi-site trials
- Calibration drift across sessions is not yet addressed
- ISO 15197 / Clarke Error Grid validation pending
- Future: temperature sensor fusion, wrist-form wearable, anemia detection via nail-bed CNN

---

## Citation

If you use SugarSync in your research, please cite:

```bibtex
@inproceedings{sajidha2024sugarsync,
  title     = {Non-Invasive Glucose Monitoring and Prediction via Near Infrared Sensor and Machine Learning},
  author    = {Sajidha, S.A. and Patnaik, Shriyansh and Tripathi, Ananya and Shrivastava, Isha},
  booktitle = {IEEE Conference Proceedings},
  year      = {2024},
  institution = {SCOPE, Vellore Institute of Technology, Chennai, India}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

<div align="center">
  <sub>Built with 💉-free curiosity at VIT Chennai · SCOPE Department</sub>
</div>
