"""
SugarSync — Local Monitoring Dashboard
Flask app serving real-time glucose predictions and signal visualisation.

Usage:
    python dashboard/app.py
    Open: http://localhost:5000
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)
cfg = load_config()
app = Flask(__name__)

CLASS_NAMES = cfg["models"]["glucose_classes"]["names"]

# ── Model Loading ──────────────────────────────────────────────────────────

_regressor  = None
_classifier = None

def get_models():
    global _regressor, _classifier
    if _regressor is None:
        model_dir = Path(cfg["paths"]["models"])
        reg_path  = model_dir / "xgboost_regressor.pkl"
        clf_path  = model_dir / "lightgbm_classifier.pkl"
        if reg_path.exists() and clf_path.exists():
            _regressor  = joblib.load(reg_path)
            _classifier = joblib.load(clf_path)
            log.info("Models loaded for dashboard.")
    return _regressor, _classifier


# ── Demo Data Generator ────────────────────────────────────────────────────

_demo_state = {
    "base_glucose": 105.0,
    "trend":        0.0,
    "samples":      [],
}

def _generate_demo_reading() -> dict:
    """Simulate realistic glucose drift for demo mode (no hardware required)."""
    state = _demo_state
    # Random walk with mean reversion
    state["trend"]    += random.gauss(0, 0.5)
    state["trend"]     = max(-5, min(5, state["trend"]))
    state["base_glucose"] += state["trend"]
    state["base_glucose"]  = max(70, min(280, state["base_glucose"]))

    glucose  = state["base_glucose"] + random.gauss(0, 2)
    hr       = 72 + random.gauss(0, 4)
    sqi      = 0.85 + random.gauss(0, 0.05)
    sqi      = float(np.clip(sqi, 0.6, 0.99))

    if glucose < 70:
        cls = 0
    elif glucose < 140:
        cls = 1
    else:
        cls = 2

    proba = [0.05, 0.05, 0.05]
    proba[cls] = 0.90

    return {
        "timestamp":    int(time.time() * 1000),
        "glucose_mgdl": round(glucose, 1),
        "class":        CLASS_NAMES[cls],
        "class_id":     cls,
        "prob_low":     round(proba[0], 3),
        "prob_normal":  round(proba[1], 3),
        "prob_high":    round(proba[2], 3),
        "hr_bpm":       round(hr, 1),
        "sqi":          round(sqi, 3),
        "demo_mode":    True,
    }


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reading")
def get_reading():
    """Return latest glucose reading (demo or live)."""
    regressor, classifier = get_models()

    if regressor is None:
        # Demo mode — no models trained yet
        reading = _generate_demo_reading()
    else:
        reading = _generate_demo_reading()   # replace with live inference
        reading["demo_mode"] = False

    _demo_state["samples"].append({
        "t":  reading["timestamp"],
        "g":  reading["glucose_mgdl"],
        "hr": reading["hr_bpm"],
    })
    # Keep last 120 readings (10 min @ 5s interval)
    _demo_state["samples"] = _demo_state["samples"][-120:]

    return jsonify(reading)


@app.route("/api/history")
def get_history():
    """Return last N glucose readings for trend chart."""
    n = int(request.args.get("n", 60))
    samples = _demo_state["samples"][-n:]
    return jsonify(samples)


@app.route("/api/status")
def get_status():
    regressor, _ = get_models()
    return jsonify({
        "models_loaded": regressor is not None,
        "demo_mode":     regressor is None,
    })


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dash_cfg = cfg["dashboard"]
    log.info(f"Starting SugarSync dashboard at http://{dash_cfg['host']}:{dash_cfg['port']}")
    app.run(
        host  = dash_cfg["host"],
        port  = dash_cfg["port"],
        debug = dash_cfg["debug"],
    )
