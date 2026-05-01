"""
SugarSync — LIME Explainability
Local instance-level explanations using surrogate linear models.

Usage:
    python src/explainability/lime_analysis.py \
        --model models/lightgbm_classifier.pkl \
        --data  data/processed/features.csv \
        --index 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime import lime_tabular

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)
cfg = load_config()

TARGET      = "glucose_mg_dl"
CLASS_NAMES = cfg["models"]["glucose_classes"]["names"]


def load_artifacts(model_path: str, data_path: str):
    model = joblib.load(model_path)
    df    = pd.read_csv(data_path)
    X     = df.drop(columns=[c for c in [TARGET, "glucose_class"] if c in df.columns])
    X     = X.select_dtypes(include=[np.number])
    return model, X


def build_explainer(X: pd.DataFrame, model, is_classifier: bool = True):
    mode = "classification" if is_classifier else "regression"
    explainer = lime_tabular.LimeTabularExplainer(
        training_data  = X.values,
        feature_names  = list(X.columns),
        class_names    = CLASS_NAMES if is_classifier else None,
        mode           = mode,
        discretize_continuous = True,
        random_state   = cfg["models"]["random_state"],
    )
    return explainer


def explain_instance(
    explainer,
    model,
    X: pd.DataFrame,
    index: int,
    num_features: int,
    output_dir: Path,
    tag: str = "",
) -> None:
    instance = X.iloc[index].values

    predict_fn = (
        model.predict_proba
        if hasattr(model, "predict_proba")
        else model.predict
    )

    explanation = explainer.explain_instance(
        data_row        = instance,
        predict_fn      = predict_fn,
        num_features    = num_features,
        num_samples     = cfg["explainability"]["lime"]["num_samples"],
    )

    # Plot
    fig = explanation.as_pyplot_figure()
    fig.patch.set_facecolor("#0f0f14")
    plt.title(f"LIME Explanation — Sample {index} ({tag})", color="#e0e0e0", pad=12)
    out = output_dir / f"lime_{tag}_idx{index}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f14")
    plt.close(fig)
    log.info(f"Saved LIME plot → {out}")

    # Text summary
    print(f"\nLIME Explanation for sample {index}:")
    for feat, weight in explanation.as_list():
        direction = "▲" if weight > 0 else "▼"
        print(f"  {direction} {feat:<40} {weight:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="LIME Analysis for SugarSync")
    parser.add_argument("--model",    required=True)
    parser.add_argument("--data",     required=True)
    parser.add_argument("--index",    type=int, default=0,
                        help="Sample index to explain")
    parser.add_argument("--n-features", type=int,
                        default=cfg["explainability"]["lime"]["num_features"])
    parser.add_argument("--regression", action="store_true",
                        help="Use regression mode (default: classification)")
    args = parser.parse_args()

    output_dir = Path(cfg["paths"]["reports"]) / "lime"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, X = load_artifacts(args.model, args.data)
    tag = Path(args.model).stem
    is_clf = not args.regression

    explainer = build_explainer(X, model, is_classifier=is_clf)
    explain_instance(
        explainer, model, X,
        index=args.index,
        num_features=args.n_features,
        output_dir=output_dir,
        tag=tag,
    )


if __name__ == "__main__":
    main()
