"""
SugarSync — Inference
Load trained models and predict glucose from a feature CSV or live PPG segment.

Usage:
    # From pre-extracted features:
    python src/models/predict.py --input data/processed/new_session.csv

    # From raw signal (runs full pipeline):
    python src/models/predict.py --raw data/raw/new_session.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.features.feature_pipeline import batch_extract

log     = get_logger(__name__)
console = Console()
cfg     = load_config()

CLASS_NAMES = cfg["models"]["glucose_classes"]["names"]
BOUNDS      = cfg["models"]["glucose_classes"]
TARGET      = "glucose_mg_dl"


# ── Model Loading ──────────────────────────────────────────────────────────

def load_models():
    model_dir = Path(cfg["paths"]["models"])
    reg_path  = model_dir / "xgboost_regressor.pkl"
    clf_path  = model_dir / "lightgbm_classifier.pkl"

    if not reg_path.exists():
        raise FileNotFoundError(
            f"Regressor not found at {reg_path}. Run train_xgboost.py first."
        )
    if not clf_path.exists():
        raise FileNotFoundError(
            f"Classifier not found at {clf_path}. Run train_lightgbm.py first."
        )

    regressor  = joblib.load(reg_path)
    classifier = joblib.load(clf_path)
    log.info("Models loaded.")
    return regressor, classifier


# ── Risk Colour ────────────────────────────────────────────────────────────

def _risk_style(cls_label: str) -> str:
    return {"Low": "bold yellow", "Normal": "bold green", "High": "bold red"}.get(
        cls_label, "white"
    )


# ── Inference on Feature DataFrame ────────────────────────────────────────

def predict_from_features(
    X: pd.DataFrame,
    regressor,
    classifier,
) -> pd.DataFrame:
    drop_cols = [c for c in [TARGET, "glucose_class"] if c in X.columns]
    X_clean   = X.drop(columns=drop_cols).select_dtypes(include=[np.number])

    glucose_pred = regressor.predict(X_clean)
    class_pred   = classifier.predict(X_clean)
    class_proba  = classifier.predict_proba(X_clean)

    result = X_clean.copy()
    result["predicted_glucose_mgdl"] = glucose_pred
    result["predicted_class"]        = [CLASS_NAMES[c] for c in class_pred]
    result["prob_low"]    = class_proba[:, 0]
    result["prob_normal"] = class_proba[:, 1]
    result["prob_high"]   = class_proba[:, 2]

    if TARGET in X.columns:
        result["actual_glucose_mgdl"] = X[TARGET].values

    return result


# ── Pretty Output ──────────────────────────────────────────────────────────

def print_summary(results: pd.DataFrame) -> None:
    table = Table(title="SugarSync — Glucose Predictions", show_lines=True)
    table.add_column("#",           style="dim",       justify="right")
    table.add_column("Glucose",     style="bold cyan", justify="center")
    table.add_column("Category",    justify="center")
    table.add_column("P(Low)",      justify="right")
    table.add_column("P(Normal)",   justify="right")
    table.add_column("P(High)",     justify="right")

    for i, row in results.iterrows():
        cls  = row["predicted_class"]
        glc  = f"{row['predicted_glucose_mgdl']:.1f} mg/dL"
        table.add_row(
            str(i),
            glc,
            f"[{_risk_style(cls)}]{cls}[/{_risk_style(cls)}]",
            f"{row['prob_low']:.2f}",
            f"{row['prob_normal']:.2f}",
            f"{row['prob_high']:.2f}",
        )

    console.print(table)

    mean_glc = results["predicted_glucose_mgdl"].mean()
    most_common = results["predicted_class"].mode()[0]
    body = (
        f"[bold]Session Mean Glucose:[/bold] {mean_glc:.1f} mg/dL\n"
        f"[bold]Predominant Category:[/bold] "
        f"[{_risk_style(most_common)}]{most_common}[/{_risk_style(most_common)}]"
    )
    console.print(Panel(body, title="Session Summary", border_style="cyan"))


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SugarSync — Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to extracted features CSV")
    group.add_argument("--raw",   help="Path to raw signal XLSX (runs full pipeline)")
    parser.add_argument("--output", default=None, help="Save predictions to CSV")
    args = parser.parse_args()

    regressor, classifier = load_models()

    if args.input:
        X = pd.read_csv(args.input)
    else:
        raw_df = pd.read_excel(args.raw)
        console.print("[cyan]Running feature extraction pipeline …[/cyan]")
        X = batch_extract(raw_df)

    if X.empty:
        console.print("[red]No valid feature vectors extracted. Check signal quality.[/red]")
        return

    results = predict_from_features(X, regressor, classifier)
    print_summary(results)

    if args.output:
        out = Path(args.output)
        results.to_csv(out, index=False)
        console.print(f"\n[green]Predictions saved → {out}[/green]")


if __name__ == "__main__":
    main()
