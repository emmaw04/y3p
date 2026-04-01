from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# ============================================================
# Hardcoded paths / settings
# ============================================================

OOF_PATH = Path("runs/final_run/stage2_multiclass/artifacts/oof_predictions.csv")
OUT_SUMMARY_PATH = Path("runs/final_run/stage2_multiclass/artifacts/base_vs_meta_summary.csv")
OUT_PER_FOLD_PATH = Path("runs/final_run/stage2_multiclass/artifacts/base_vs_meta_per_fold.csv")

# Keep this order aligned with your c0..c4 probability columns
CLASS_LABELS = ["HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET"]

MODELS: Dict[str, List[str]] = {
    "rf": [f"p_rf_c{i}" for i in range(len(CLASS_LABELS))],
    "xgb": [f"p_xgb_c{i}" for i in range(len(CLASS_LABELS))],
    "svm": [f"p_svm_c{i}" for i in range(len(CLASS_LABELS))],
    "tcn_gru": [f"p_tcn_gru_c{i}" for i in range(len(CLASS_LABELS))],
    "meta": [f"meta_proba_c{i}" for i in range(len(CLASS_LABELS))],
}


def _require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalise_rows(proba: np.ndarray) -> np.ndarray:
    """
    Safeguard against probabilities not summing exactly to 1 due to rounding.
    """
    proba = np.clip(proba, 1e-15, 1.0)
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return proba / row_sums


def _encode_targets(y: pd.Series) -> np.ndarray:
    """
    Map string class labels to integer ids using CLASS_LABELS order.
    """
    unknown = sorted(set(y.unique()) - set(CLASS_LABELS))
    if unknown:
        raise ValueError(
            f"Found labels in y_compound not present in CLASS_LABELS: {unknown}"
        )

    label_to_int = {label: i for i, label in enumerate(CLASS_LABELS)}
    return y.map(label_to_int).to_numpy()


def _safe_macro_pr_auc(y_true_int: np.ndarray, proba: np.ndarray) -> float:
    """
    Macro one-vs-rest PR-AUC. Returns NaN if not computable.
    """
    try:
        y_bin = label_binarize(y_true_int, classes=np.arange(len(CLASS_LABELS)))
        return float(average_precision_score(y_bin, proba, average="macro"))
    except Exception:
        return np.nan


def _safe_macro_roc_auc(y_true_int: np.ndarray, proba: np.ndarray) -> float:
    """
    Macro one-vs-rest ROC-AUC. Returns NaN if not computable.
    """
    try:
        return float(
            roc_auc_score(
                y_true_int,
                proba,
                multi_class="ovr",
                average="macro",
                labels=np.arange(len(CLASS_LABELS)),
            )
        )
    except Exception:
        return np.nan


def evaluate_one_split(df_split: pd.DataFrame, proba_cols: List[str]) -> Dict[str, float]:
    """
    Evaluate one model on one fold.
    """
    y_true_int = _encode_targets(df_split["y_compound"])
    proba = _normalise_rows(df_split[proba_cols].to_numpy(dtype=float))
    y_pred_int = np.argmax(proba, axis=1)

    metrics = {
        "rows": int(len(df_split)),
        "accuracy": accuracy_score(y_true_int, y_pred_int),
        "macro_f1": f1_score(y_true_int, y_pred_int, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true_int, y_pred_int, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true_int, y_pred_int, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true_int, y_pred_int, average="macro", zero_division=0),
        "logloss": log_loss(y_true_int, proba, labels=np.arange(len(CLASS_LABELS))),
        "macro_pr_auc_ovr": _safe_macro_pr_auc(y_true_int, proba),
        "macro_roc_auc_ovr": _safe_macro_roc_auc(y_true_int, proba),
    }
    return metrics


def summarise_across_folds(per_fold_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and std of each metric across folds for each model.
    """
    metric_cols = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "logloss",
        "macro_pr_auc_ovr",
        "macro_roc_auc_ovr",
    ]

    rows = []
    for model_name, g in per_fold_df.groupby("model", sort=False):
        row = {"model": model_name, "n_folds": g["fold"].nunique(), "total_rows": g["rows"].sum()}
        for col in metric_cols:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_std"] = g[col].std(ddof=1)
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Helpful ranking: higher macro_f1 is better, lower logloss is better
    if not summary.empty:
        summary = summary.sort_values(
            by=["macro_f1_mean", "logloss_mean"],
            ascending=[False, True]
        ).reset_index(drop=True)

    return summary


def main() -> None:
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"Could not find file: {OOF_PATH}")

    df = pd.read_csv(OOF_PATH)

    required = ["fold", "y_compound"]
    for cols in MODELS.values():
        required.extend(cols)
    _require_columns(df, sorted(set(required)), "OOF file")

    per_fold_rows = []

    for fold_value in sorted(df["fold"].dropna().unique()):
        df_fold = df[df["fold"] == fold_value].copy()

        for model_name, proba_cols in MODELS.items():
            metrics = evaluate_one_split(df_fold, proba_cols)
            metrics["model"] = model_name
            metrics["fold"] = fold_value
            per_fold_rows.append(metrics)

    per_fold_df = pd.DataFrame(per_fold_rows)
    summary_df = summarise_across_folds(per_fold_df)

    OUT_PER_FOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    per_fold_df.to_csv(OUT_PER_FOLD_PATH, index=False)
    summary_df.to_csv(OUT_SUMMARY_PATH, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n=== Per-fold metrics ===")
    print(per_fold_df.round(4))

    print("\n=== Mean ± std across folds ===")
    print(summary_df.round(4))

    # Small comparison block to make the value of stacking obvious
    if "meta" in summary_df["model"].values:
        meta_row = summary_df[summary_df["model"] == "meta"].iloc[0]

        print("\n=== Comparison against meta learner ===")
        for base_model in ["rf", "xgb", "svm", "tcn_gru"]:
            if base_model not in summary_df["model"].values:
                continue

            base_row = summary_df[summary_df["model"] == base_model].iloc[0]
            print(
                f"{base_model:8s} | "
                f"Δmacro_f1 = {meta_row['macro_f1_mean'] - base_row['macro_f1_mean']:+.4f} | "
                f"Δaccuracy = {meta_row['accuracy_mean'] - base_row['accuracy_mean']:+.4f} | "
                f"Δlogloss = {meta_row['logloss_mean'] - base_row['logloss_mean']:+.4f}"
            )

        print("\nInterpretation:")
        print("- Positive Δmacro_f1 / Δaccuracy means the meta learner improved over that base model.")
        print("- Negative Δlogloss means the meta learner has better probability quality (lower is better).")
        print("- For this task, macro F1 and logloss are more informative than accuracy alone.")


if __name__ == "__main__":
    main()