"""
SugarSync — LightGBM Classifier Training
Classifies glucose into Low / Normal / High categories.

Usage:
    python src/models/train_lightgbm.py --data data/processed/features.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from rich.console import Console
from rich.table import Table

from src.utils.config import load_config
from src.utils.logger import get_logger

log     = get_logger(__name__)
console = Console()
cfg     = load_config()

TARGET      = "glucose_mg_dl"
CLASS_COL   = "glucose_class"
CLASS_NAMES = cfg["models"]["glucose_classes"]["names"]   # ["Low", "Normal", "High"]
BOUNDS      = cfg["models"]["glucose_classes"]


def _assign_class(glucose: float) -> int:
    """Map continuous glucose value to class label (0=Low, 1=Normal, 2=High)."""
    low_ub  = BOUNDS["low"][1]      # 70
    norm_ub = BOUNDS["normal"][1]   # 140
    if glucose < low_ub:
        return 0
    elif glucose < norm_ub:
        return 1
    else:
        return 2


def load_dataset(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    df[CLASS_COL] = df[TARGET].apply(_assign_class)
    y = df[CLASS_COL]
    X = df.drop(columns=[TARGET, CLASS_COL]).select_dtypes(include=[np.number])

    dist = y.value_counts().to_dict()
    console.print(f"Class distribution: {dict(zip(CLASS_NAMES, [dist.get(i,0) for i in range(3)]))}")
    return X, y


def split_data(X, y, seed=42):
    splits = cfg["models"]["train_val_test_split"]
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=splits[2], random_state=seed, stratify=y
    )
    val_frac = splits[1] / (splits[0] + splits[1])
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, random_state=seed, stratify=y_tv
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_lightgbm(X_train, y_train, X_val, y_val):
    lgb_cfg = cfg["models"]["lightgbm_classifier"]

    model = lgb.LGBMClassifier(
        n_estimators          = lgb_cfg["n_estimators"],
        num_leaves            = lgb_cfg["num_leaves"],
        learning_rate         = lgb_cfg["learning_rate"],
        feature_fraction      = lgb_cfg["feature_fraction"],
        bagging_fraction      = lgb_cfg["bagging_fraction"],
        bagging_freq          = lgb_cfg["bagging_freq"],
        class_weight          = lgb_cfg["class_weight"],
        verbose               = lgb_cfg["verbose"],
        random_state          = cfg["models"]["random_state"],
        n_jobs                = -1,
    )

    callbacks = [lgb.early_stopping(lgb_cfg["early_stopping_rounds"], verbose=False)]

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    log.info(f"Best iteration: {model.best_iteration_}")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)

    # Macro-averaged one-vs-rest AUC
    y_bin  = label_binarize(y_test, classes=[0, 1, 2])
    auc    = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")

    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    table = Table(title="LightGBM — Test Performance")
    table.add_column("Metric",   style="cyan")
    table.add_column("Value",    style="bold green")
    table.add_row("Accuracy",    f"{acc:.4f}")
    table.add_row("ROC-AUC",     f"{auc:.4f}")
    console.print(table)
    console.print(f"\n{report}")
    console.print("Confusion Matrix:")
    console.print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

    return {"accuracy": acc, "roc_auc": auc, "confusion_matrix": cm.tolist()}


def cross_validate_clf(X, y, n_splits=5):
    lgb_cfg = cfg["models"]["lightgbm_classifier"]
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=cfg["models"]["random_state"])
    accs, aucs = [], []

    for fold, (tr, vl) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(
            n_estimators  = 300,
            num_leaves    = lgb_cfg["num_leaves"],
            learning_rate = lgb_cfg["learning_rate"],
            class_weight  = "balanced",
            verbose       = -1,
            random_state  = cfg["models"]["random_state"],
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        preds  = model.predict(X.iloc[vl])
        probas = model.predict_proba(X.iloc[vl])
        y_bin  = label_binarize(y.iloc[vl], classes=[0, 1, 2])

        acc = accuracy_score(y.iloc[vl], preds)
        auc = roc_auc_score(y_bin, probas, multi_class="ovr", average="macro")
        accs.append(acc); aucs.append(auc)
        console.print(f"  Fold {fold+1}: Acc={acc:.3f}, AUC={auc:.3f}")

    console.print(
        f"\n[bold]CV:[/bold] Acc = {np.mean(accs):.3f}±{np.std(accs):.3f} | "
        f"AUC = {np.mean(aucs):.3f}±{np.std(aucs):.3f}"
    )


def save_model(model, feature_names, metrics):
    model_dir = Path(cfg["paths"]["models"])
    model_dir.mkdir(parents=True, exist_ok=True)

    path = model_dir / "lightgbm_classifier.pkl"
    joblib.dump(model, path)

    meta = {
        "model": "LGBMClassifier",
        "classes": CLASS_NAMES,
        "features": feature_names,
        "best_iteration": int(model.best_iteration_),
        **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
    }
    with open(model_dir / "lightgbm_classifier_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Classifier saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    required=True)
    parser.add_argument("--cv",      action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    X, y = load_dataset(args.data)

    if args.cv:
        console.print("\n[bold cyan]5-Fold Stratified CV …[/bold cyan]")
        cross_validate_clf(X, y)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    console.print("\n[bold cyan]Training LightGBM Classifier …[/bold cyan]")
    model = train_lightgbm(X_train, y_train, X_val, y_val)

    metrics = evaluate_model(model, X_test, y_test)

    if not args.no_save:
        save_model(model, list(X.columns), metrics)
        console.print("\n[bold green]✓ Classifier saved.[/bold green]")


if __name__ == "__main__":
    main()
