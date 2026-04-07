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

#hardcoded paths
STAGE1_PATH = Path("runs/final_run/stage1_binary/artifacts/oof_predictions.csv")
STAGE2_PATH = Path("runs/final_run/stage2_multiclass/artifacts/oof_predictions.csv")
OUT_PATH = Path("runs/final_run/oof_slice_metrics.csv")

#hardcoded decision rules
PIT_THRESHOLD = 0.264
COMPOUND_LABELS = ["HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET"]
META_COLS_STAGE2 = [f"meta_proba_c{i}" for i in range(len(COMPOUND_LABELS))] #the probability columns expected in the stage 2 OOF probabilities file
GP_COL = "race_track"

#threshold chosen earlier >= 6 appearances across 8 seasons = more seen
MORE_SEEN_GPS = {
    "Abu Dhabi Grand Prix",
    "Austrian Grand Prix",
    "Bahrain Grand Prix",
    "Belgian Grand Prix",
    "British Grand Prix",
    "Hungarian Grand Prix",
    "Italian Grand Prix",
    "Spanish Grand Prix",
    "Azerbaijan Grand Prix",
    "Monaco Grand Prix",
    "United States Grand Prix",
    "Australian Grand Prix",
    "Canadian Grand Prix",
    "Japanese Grand Prix",
    "Singapore Grand Prix",
}

LESS_SEEN_GPS = {
    "Dutch Grand Prix",
    "Emilia Romagna Grand Prix",
    "Mexico City Grand Prix",
    "Saudi Arabian Grand Prix",
    "São Paulo Grand Prix",
    "Chinese Grand Prix",
    "French Grand Prix",
    "Miami Grand Prix",
    "Qatar Grand Prix",
    "Russian Grand Prix",
    "Las Vegas Grand Prix",
    "Brazilian Grand Prix",
    "German Grand Prix",
    "Mexican Grand Prix",
    "Portuguese Grand Prix",
    "Styrian Grand Prix",
    "Turkish Grand Prix",
    "70th Anniversary Grand Prix",
    "Eifel Grand Prix",
    "Sakhir Grand Prix",
    "Tuscan Grand Prix",
}

def _require_columns(df: pd.DataFrame, cols: list[str], name: str):
    """
    checks the dataframe has all the columns the script needs
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

def _race_slices(stage1: pd.DataFrame):
    """
    builds a dictionary of each slice and the race ids of races that belong to that slice
    """
    race_ids = set(stage1["race_id"].unique())
    wet_races = set(stage1.loc[stage1["is_wet_race"] == 1, "race_id"].unique())
    fcy_races = set(stage1.loc[stage1["fcy_status"] != 0, "race_id"].unique()) #finds all rows where there was at least one row with non zero fcy status

    more_seen_races = set(stage1.loc[stage1[GP_COL].isin(MORE_SEEN_GPS), "race_id"].unique())
    less_seen_races = set(stage1.loc[stage1[GP_COL].isin(LESS_SEEN_GPS), "race_id"].unique())

    return {
        "all_races": race_ids,
        "wet_races": wet_races,
        "dry_races": race_ids - wet_races,
        "races_with_fcy": fcy_races,
        "races_without_fcy": race_ids - fcy_races,
        "more_seen_tracks": more_seen_races,
        "less_seen_tracks": less_seen_races,
    }

def _binary_metrics(df: pd.DataFrame):
    """
    computes metrics for stage 1 of the model
    """
    y_true = df["y_pit"].astype(int).to_numpy()
    y_score = df["meta_proba"].astype(float).to_numpy()
    y_pred = (y_score >= PIT_THRESHOLD).astype(int) #converts probs to hard predictions
    proba_2col = np.column_stack([1.0 - y_score, y_score])

    return {
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "positives": int(y_true.sum()),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "logloss": float(log_loss(y_true, proba_2col, labels=[0, 1])),
    }


def _multiclass_metrics(df: pd.DataFrame):
    """
    computes metrics for stage 2 of the model
    """
    label_to_idx = {label: i for i, label in enumerate(COMPOUND_LABELS)}
    y_true = df["y_compound"].map(label_to_idx).astype(int).to_numpy()
    y_score = df[META_COLS_STAGE2].astype(float).to_numpy()
    y_pred = y_score.argmax(axis=1)
    present = np.unique(y_true)

    return {
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "pr_auc": float(
        np.mean([average_precision_score((y_true == cls).astype(int), y_score[:, cls])for cls in present])),
        "logloss": float(log_loss(y_true, y_score, labels=list(range(len(COMPOUND_LABELS))))),
    }

def _summarise(stage_name: str, df: pd.DataFrame, slices: dict[str, set[int]], metric_fn):
    """
    computes metrics for each slice
    """
    rows = []
    for slice_name, race_ids in slices.items():
        subset = df[df["race_id"].isin(race_ids)].copy()
        if subset.empty:
            continue
        row = {"stage": stage_name, "slice": slice_name}
        row.update(metric_fn(subset))
        rows.append(row)
    return pd.DataFrame(rows)

def main():
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