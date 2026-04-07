# src/utils/tune_meta_thresholds.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

PREDS_PATH = "runs/final_run/stage1_binary/artifacts/oof_predictions.csv"
OUT_PATH = "runs/final_run/stage1_binary/artifacts/meta_threshold.json"
GRID_SIZE = 1001

def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float):
    """
    computes evaluation metrics at a given threshold
    """
    y_pred = (y_score >= thr).astype(int)
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def sweep_best_f1(y_true: np.ndarray, y_score: np.ndarray, grid_size: int = 1001):
    """
    searches for the best threshold
    """
    thresholds = np.linspace(0.0, 1.0, grid_size)
    best = None

    #loop over every threshold
    for thr in thresholds:
        metrics = metrics_at_threshold(y_true, y_score, thr)
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics

    return best

def main():
    df = pd.read_csv(PREDS_PATH) #load the csv
    y_true = df["y_pit"].to_numpy() #get true labels and predicted probabilities
    y_score = df["meta_proba"].to_numpy()

    #create a dictionary to store results compared to default threshold of 0.5
    results = {
        "metrics_at_0.5": metrics_at_threshold(y_true, y_score, 0.5),
        "best_threshold_f1": sweep_best_f1(y_true, y_score, GRID_SIZE),
    }
    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()