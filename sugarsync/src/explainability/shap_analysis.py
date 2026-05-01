"""
SugarSync — SHAP Explainability
Global and local feature importance using Shapley Additive Explanations.

Usage:
    python src/explainability/shap_analysis.py \
        --model models/xgboost_regressor.pkl \
        --data  data/processed/features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)
cfg = load_config()

TARGET = "glucose_mg_dl"

PLOT_STYLE = {
    "font.family":    "monospace",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "#0f0f14",
    "axes.facecolor":     "#0f0f14",
    "text.color":         "#e0e0e0",
    "axes.labelcolor":    "#e0e0e0",
    "xtick.color":        "#a0a0a0",
    "ytick.color":        "#a0a0a0",
}


def load_artifacts(model_path: str, data_path: str):
    model = joblib.load(model_path)
    df    = pd.read_csv(data_path)
    y     = df.get(TARGET)
    X     = df.drop(columns=[c for c in [TARGET, "glucose_class"] if c in df.columns])
    X     = X.select_dtypes(include=[np.number])
    return model, X, y


def compute_shap_values(model, X: pd.DataFrame, background_n: int = 100):
    """
    Compute SHAP values using TreeExplainer.

    TreeExplainer is exact (not approximate) for tree-based models.
    Background dataset is a random sample of the training data.
    """
    background = shap.sample(X, min(background_n, len(X)))
    explainer  = shap.TreeExplainer(model, background)
    shap_vals  = explainer(X)
    log.info(f"Computed SHAP values for {len(X)} samples × {X.shape[1]} features.")
    return explainer, shap_vals


def plot_summary_beeswarm(shap_vals, X: pd.DataFrame, output_dir: Path, tag: str = ""):
    """Global SHAP summary: beeswarm plot coloured by feature value."""
    plt.rcParams.update(PLOT_STYLE)
    shap.plots.beeswarm(shap_vals, max_display=20, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor("#0f0f14")
    plt.title(f"SHAP Feature Importance — {tag}", color="#e0e0e0", pad=12)
    out = output_dir / f"shap_beeswarm_{tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f14")
    plt.close()
    log.info(f"Saved beeswarm → {out}")


def plot_bar_importance(shap_vals, X: pd.DataFrame, output_dir: Path, tag: str = ""):
    """Mean |SHAP| bar chart — global feature ranking."""
    mean_abs = np.abs(shap_vals.values).mean(axis=0)
    feat_imp = pd.Series(mean_abs, index=X.columns).nlargest(20)

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f0f14")
    ax.set_facecolor("#0f0f14")

    colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(feat_imp)))
    ax.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP value|", color="#e0e0e0")
    ax.set_title(f"Global Feature Importance — {tag}", color="#e0e0e0")
    plt.tight_layout()

    out = output_dir / f"shap_importance_{tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f14")
    plt.close()
    log.info(f"Saved bar chart → {out}")


def plot_waterfall(shap_vals, X: pd.DataFrame, index: int,
                   output_dir: Path, tag: str = ""):
    """Local explanation for a single sample."""
    plt.rcParams.update(PLOT_STYLE)
    shap.plots.waterfall(shap_vals[index], show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor("#0f0f14")
    plt.title(f"Local SHAP — Sample {index} ({tag})", color="#e0e0e0")
    out = output_dir / f"shap_waterfall_{tag}_idx{index}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f14")
    plt.close()
    log.info(f"Saved waterfall → {out}")


def save_importance_csv(shap_vals, X: pd.DataFrame, output_dir: Path, tag: str = ""):
    """Export mean |SHAP| per feature as CSV for downstream analysis."""
    mean_abs = np.abs(shap_vals.values).mean(axis=0)
    df = pd.DataFrame({
        "feature":       X.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    out = output_dir / f"shap_importance_{tag}.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved importance CSV → {out}")
    return df


def run_shap_analysis(model_path: str, data_path: str) -> None:
    output_dir = Path(cfg["paths"]["reports"]) / "shap"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, X, y = load_artifacts(model_path, data_path)
    model_name  = Path(model_path).stem

    console_log = f"Running SHAP for model: {model_name}"
    log.info(console_log)
    print(console_log)

    explainer, shap_vals = compute_shap_values(
        model, X, cfg["explainability"]["shap"]["background_samples"]
    )

    plot_summary_beeswarm(shap_vals, X, output_dir, model_name)
    plot_bar_importance(shap_vals,   X, output_dir, model_name)
    plot_waterfall(shap_vals, X, index=0, output_dir=output_dir, tag=model_name)
    imp_df = save_importance_csv(shap_vals, X, output_dir, model_name)

    print(f"\nTop 10 features by mean |SHAP|:")
    print(imp_df.head(10).to_string(index=False))
    print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="SHAP Analysis for SugarSync models")
    parser.add_argument("--model", required=True, help="Path to .pkl model file")
    parser.add_argument("--data",  required=True, help="Path to features CSV")
    args = parser.parse_args()

    run_shap_analysis(args.model, args.data)


if __name__ == "__main__":
    main()
