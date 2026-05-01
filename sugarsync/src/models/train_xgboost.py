"""
SugarSync — XGBoost Regressor Training
Trains a glucose regression model with L1/L2 regularization, SHAP support,
early stopping, and cross-validation.

Usage:
    python src/models/train_xgboost.py --data data/processed/features.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from rich.console import Console
from rich.table import Table

from src.utils.config import load_config
from src.utils.logger import get_logger

log     = get_logger(__name__)
console = Console()
cfg     = load_config()

TARGET = "glucose_mg_dl"


# ── Data Loading ───────────────────────────────────────────────────────────

def load_dataset(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in {path}")

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # Drop any string columns (shouldn't exist after feature extraction)
    X = X.select_dtypes(include=[np.number])

    log.info(f"Loaded {len(X)} samples × {X.shape[1]} features from {path}")
    return X, y


# ── Train / Val / Test Split ───────────────────────────────────────────────

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
):
    model_cfg = cfg["models"]
    train_r, val_r, test_r = model_cfg["train_val_test_split"]

    # First pull out test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_r, random_state=seed, shuffle=True
    )
    # Then split train/val from trainval
    val_fraction = val_r / (train_r + val_r)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction, random_state=seed
    )

    log.info(
        f"Split → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Model Training ─────────────────────────────────────────────────────────

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> xgb.XGBRegressor:
    xgb_cfg = cfg["models"]["xgboost_regressor"]

    model = xgb.XGBRegressor(
        n_estimators          = xgb_cfg["n_estimators"],
        max_depth             = xgb_cfg["max_depth"],
        learning_rate         = xgb_cfg["learning_rate"],
        subsample             = xgb_cfg["subsample"],
        colsample_bytree      = xgb_cfg["colsample_bytree"],
        reg_alpha             = xgb_cfg["reg_alpha"],
        reg_lambda            = xgb_cfg["reg_lambda"],
        early_stopping_rounds = xgb_cfg["early_stopping_rounds"],
        eval_metric           = xgb_cfg["eval_metric"],
        verbosity             = xgb_cfg["verbosity"],
        random_state          = cfg["models"]["random_state"],
        n_jobs                = -1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    log.info(
        f"Best iteration: {model.best_iteration} | "
        f"Val MAE: {model.best_score:.4f}"
    )
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    split_name: str = "Test",
) -> dict:
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    metrics = {"split": split_name, "MAE": mae, "RMSE": rmse, "R2": r2}

    table = Table(title=f"XGBoost — {split_name} Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="bold green")
    table.add_row("MAE",  f"{mae:.4f} mg/dL")
    table.add_row("RMSE", f"{rmse:.4f} mg/dL")
    table.add_row("R²",   f"{r2:.4f}")
    console.print(table)

    return metrics


# ── Cross-Validation ───────────────────────────────────────────────────────

def cross_validate(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """5-fold CV to report generalisation error."""
    xgb_cfg = cfg["models"]["xgboost_regressor"]
    kf      = KFold(n_splits=n_splits, shuffle=True,
                    random_state=cfg["models"]["random_state"])

    maes, rmses, r2s = [], [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(
            n_estimators   = 300,
            max_depth      = xgb_cfg["max_depth"],
            learning_rate  = xgb_cfg["learning_rate"],
            subsample      = xgb_cfg["subsample"],
            colsample_bytree = xgb_cfg["colsample_bytree"],
            reg_alpha      = xgb_cfg["reg_alpha"],
            reg_lambda     = xgb_cfg["reg_lambda"],
            verbosity      = 0,
            random_state   = cfg["models"]["random_state"],
            n_jobs         = -1,
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_vl)
        maes.append(mean_absolute_error(y_vl, preds))
        rmses.append(np.sqrt(mean_squared_error(y_vl, preds)))
        r2s.append(r2_score(y_vl, preds))
        console.print(f"  Fold {fold+1}: MAE={maes[-1]:.3f}, R²={r2s[-1]:.3f}")

    results = {
        "cv_mae_mean":  float(np.mean(maes)),
        "cv_mae_std":   float(np.std(maes)),
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_r2_mean":   float(np.mean(r2s)),
    }
    console.print(
        f"\n[bold]CV Results:[/bold] MAE = {results['cv_mae_mean']:.3f} "
        f"± {results['cv_mae_std']:.3f} | R² = {results['cv_r2_mean']:.3f}"
    )
    return results


# ── Save Artefacts ─────────────────────────────────────────────────────────

def save_model(model: xgb.XGBRegressor, feature_names: list, metrics: dict) -> None:
    model_dir = Path(cfg["paths"]["models"])
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "xgboost_regressor.pkl"
    joblib.dump(model, model_path)
    log.info(f"Model saved → {model_path}")

    meta = {
        "model": "XGBoostRegressor",
        "target": TARGET,
        "features": feature_names,
        "best_iteration": int(model.best_iteration),
        **metrics,
    }
    meta_path = model_dir / "xgboost_regressor_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata saved → {meta_path}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost glucose regressor"
    )
    parser.add_argument("--data",    required=True, help="Path to features CSV")
    parser.add_argument("--cv",      action="store_true", help="Run 5-fold CV")
    parser.add_argument("--no-save", action="store_true", help="Skip model saving")
    args = parser.parse_args()

    X, y = load_dataset(args.data)

    if args.cv:
        console.print("\n[bold cyan]Running 5-Fold Cross-Validation …[/bold cyan]")
        cross_validate(X, y)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    console.print("\n[bold cyan]Training XGBoost Regressor …[/bold cyan]")
    model = train_xgboost(X_train, y_train, X_val, y_val)

    val_metrics  = evaluate_model(model, X_val,  y_val,  "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")

    all_metrics = {**val_metrics, **{f"test_{k}": v for k, v in test_metrics.items()}}

    if not args.no_save:
        save_model(model, list(X.columns), all_metrics)
        console.print("\n[bold green]✓ Model and metadata saved.[/bold green]")


if __name__ == "__main__":
    main()
