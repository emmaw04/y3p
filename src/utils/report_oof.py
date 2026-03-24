from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

# Hardcoded paths
STAGE1_PATH = Path("runs/final_run/stage1_binary/artifacts/oof_predictions.csv")
STAGE2_PATH = Path("runs/final_run/stage2_multiclass/artifacts/oof_predictions.csv")
OUT_PATH = Path("runs/final_run/oof_slice_metrics.csv")

# Hardcoded decision rules
PIT_THRESHOLD = 0.3120
COMPOUND_LABELS = ["HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET"]
META_COLS_STAGE2 = [f"meta_proba_c{i}" for i in range(len(COMPOUND_LABELS))]


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _race_slices(stage1: pd.DataFrame) -> dict[str, set[int]]:
    race_ids = set(stage1["race_id"].unique())
    wet_races = set(stage1.loc[stage1["is_wet_race"] == 1, "race_id"].unique())
    fcy_races = set(stage1.loc[stage1["fcy_status"] != 0, "race_id"].unique())
    return {
        "all_races": race_ids,
        "wet_races": wet_races,
        "dry_races": race_ids - wet_races,
        "races_with_fcy": fcy_races,
        "races_without_fcy": race_ids - fcy_races,
    }


def _binary_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["y_pit"].astype(int).to_numpy()
    y_score = df["meta_proba"].astype(float).to_numpy()
    y_pred = (y_score >= PIT_THRESHOLD).astype(int)

    pr_auc = np.nan
    if np.unique(y_true).size > 1 and y_true.sum() > 0:
        pr_auc = float(average_precision_score(y_true, y_score))

    proba_2col = np.column_stack([1.0 - y_score, y_score])

    return {
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "positives": int(y_true.sum()),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "pr_auc": pr_auc,
        "logloss": float(log_loss(y_true, proba_2col, labels=[0, 1])),
    }


def _multiclass_metrics(df: pd.DataFrame) -> dict[str, float]:
    label_to_idx = {label: i for i, label in enumerate(COMPOUND_LABELS)}
    bad = sorted(set(df["y_compound"].dropna().unique()) - set(COMPOUND_LABELS))
    if bad:
        raise ValueError(
            "Unexpected y_compound labels found. Update COMPOUND_LABELS to match training order: "
            f"{bad}"
        )

    y_true = df["y_compound"].map(label_to_idx).astype(int).to_numpy()
    y_score = df[META_COLS_STAGE2].astype(float).to_numpy()
    y_pred = y_score.argmax(axis=1)

    present = np.unique(y_true)
    pr_auc = float(
        np.mean([
            average_precision_score((y_true == cls).astype(int), y_score[:, cls])
            for cls in present
        ])
    )

    return {
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "pr_auc": pr_auc,
        "logloss": float(log_loss(y_true, y_score, labels=list(range(len(COMPOUND_LABELS))))),
    }


def _summarise(stage_name: str, df: pd.DataFrame, slices: dict[str, set[int]], metric_fn) -> pd.DataFrame:
    rows = []
    for slice_name, race_ids in slices.items():
        subset = df[df["race_id"].isin(race_ids)].copy()
        if subset.empty:
            continue
        row = {"stage": stage_name, "slice": slice_name}
        row.update(metric_fn(subset))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    stage1 = pd.read_csv(STAGE1_PATH)
    stage2 = pd.read_csv(STAGE2_PATH)

    _require_columns(stage1, ["race_id", "y_pit", "meta_proba", "is_wet_race", "fcy_status"], "stage1")
    _require_columns(stage2, ["race_id", "y_compound", *META_COLS_STAGE2], "stage2")

    slices = _race_slices(stage1)

    stage1_report = _summarise("stage1_binary", stage1, slices, _binary_metrics)
    stage2_report = _summarise("stage2_multiclass", stage2, slices, _multiclass_metrics)
    report = pd.concat([stage1_report, stage2_report], ignore_index=True)

    metric_cols = ["f1", "precision", "recall", "pr_auc", "logloss"]
    report[metric_cols] = report[metric_cols].round(4)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT_PATH, index=False)

    pd.set_option("display.max_columns", None)
    print(report.to_string(index=False))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
