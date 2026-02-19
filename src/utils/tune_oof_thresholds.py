# src/utils/tune_oof_thresholds.py
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# sklearn metrics
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# -----------------------------
# loading helpers
# -----------------------------
def _load_stage1_df(data_stage1_path: str):
    """
    Load the stage1 dataframe using the project loader (so any filtering/processing
    matches training), then defensively remove HOLDOUT_RACE_IDS again.
    """
    try:
        from src.data.data import load_stage1_dataset, HOLDOUT_RACE_IDS  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import src.data.data.load_stage1_dataset/HOLDOUT_RACE_IDS.\n"
            "Run from repo root, e.g.:\n"
            "  python -m src.utils.tune_oof_thresholds --data_stage1 ... --model ...\n"
        ) from e

    # Be robust to different signatures across your code versions.
    df = None
    errors: List[str] = []

    # 1) most common: load_stage1_dataset(path)
    try:
        df = load_stage1_dataset(data_stage1_path)
    except Exception as e:
        errors.append(f"load_stage1_dataset(path) -> {type(e).__name__}: {e}")

    # 2) some variants ignore args / take none
    if df is None:
        try:
            df = load_stage1_dataset()
        except Exception as e:
            errors.append(f"load_stage1_dataset() -> {type(e).__name__}: {e}")

    # 3) some variants accept strict=
    if df is None:
        try:
            df = load_stage1_dataset(data_stage1_path, strict=True)
        except Exception as e:
            errors.append(f"load_stage1_dataset(path, strict=True) -> {type(e).__name__}: {e}")

    if df is None:
        raise RuntimeError(
            "Failed to load stage1 dataset via load_stage1_dataset with several signatures.\n"
            + "\n".join(errors)
        )

    # Defensive holdout filtering (preserves relative order)
    if "race_id" in df.columns:
        df = df.loc[~df["race_id"].isin(set(HOLDOUT_RACE_IDS))].reset_index(drop=True)

    return df


def _load_npy_1d(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing .npy file: {p}")
    arr = np.load(p)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(float)


def _infer_label_col(columns: List[str]) -> Optional[str]:
    """
    Try common label names used in binary pit-stop datasets.
    Override with --label_col if needed.
    """
    return "y_pit"


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


# -----------------------------
# threshold sweep
# -----------------------------
def _metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_score >= thr).astype(int)

    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _sweep_best_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    grid_size: int = 1001,
) -> Tuple[float, Dict[str, float]]:
    """
    Sweep thresholds from 0 to 1 inclusive and pick threshold maximizing F1.
    Returns (best_thr, best_metrics_dict).
    """
    thresholds = np.linspace(0.0, 1.0, grid_size, dtype=float)

    best_thr = 0.5
    best = {"f1": -1.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "threshold": 0.5}

    # brute-force sweep (fast enough for ~100k rows and ~1000 thresholds)
    for t in thresholds:
        m = _metrics_at_threshold(y_true, y_score, float(t))
        # break ties by preferring slightly higher recall (useful for rare-event detection)
        if (m["f1"] > best["f1"]) or (np.isclose(m["f1"], best["f1"]) and m["recall"] > best["recall"]):
            best = m
            best_thr = float(t)

    return best_thr, best


# -----------------------------
# reporting + saving
# -----------------------------
@dataclass
class ModelThresholdReport:
    model: str
    npy_path: str
    n_rows_df: int
    n_rows_used: int
    pos_rate_used: float

    pr_auc: Optional[float]
    roc_auc: Optional[float]

    metrics_t05: Dict[str, float]
    best_threshold: float
    metrics_best: Dict[str, float]


def _safe_auc(fn, y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """
    AUC requires both classes present; return None if not computable.
    """
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(fn(y_true, y_score))
    except Exception:
        return None


def _write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    import csv

    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Tune per-model thresholds on OOF probabilities to maximize F1.\n"
            "Loads stage1 dataset via load_stage1_dataset (holdouts removed) and aligns by row order."
        )
    )
    ap.add_argument(
        "--data_stage1",
        required=True,
        help="Path to the stage1 CSV (or whatever your loader expects). Will be loaded via load_stage1_dataset.",
    )
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec as name=path_to_oof_pred_proba_pos.npy (repeatable).",
    )
    ap.add_argument(
        "--label_col",
        default="",
        help="Label column in the stage1 dataframe. If omitted, we try to infer it.",
    )
    ap.add_argument(
        "--grid_size",
        type=int,
        default=1001,
        help="Number of thresholds in [0,1] to sweep (default 1001 => step 0.001).",
    )
    ap.add_argument(
        "--baseline_threshold",
        type=float,
        default=0.5,
        help="Baseline threshold to report as 'before' (default 0.5).",
    )
    ap.add_argument(
        "--out_dir",
        default="results/runs/seq/stage1/binary/threshold_tuning",
        help="Directory to save CSV/JSON outputs.",
    )
    args = ap.parse_args()

    # parse model specs
    model_paths: Dict[str, str] = {}
    for spec in args.model:
        if "=" not in spec:
            print(f"[ERROR] Bad --model spec '{spec}'. Use name=path.npy", file=sys.stderr)
            sys.exit(1)
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            print(f"[ERROR] Empty model name in spec '{spec}'", file=sys.stderr)
            sys.exit(1)
        model_paths[name] = path

    # load stage1 df (filters holdouts inside loader; then we defensively filter again)
    try:
        df = _load_stage1_df(args.data_stage1)
    except Exception as e:
        print(f"[ERROR] Failed loading stage1 dataset via load_stage1_dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # label col
    label_col = args.label_col.strip() or _infer_label_col(list(df.columns))
    if not label_col or label_col not in df.columns:
        print("[ERROR] Could not determine label column.", file=sys.stderr)
        print("  Pass --label_col YOUR_LABEL_COL", file=sys.stderr)
        print(f"  Columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    y = df[label_col].to_numpy()
    # ensure binary ints
    y = np.asarray(y).astype(int)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: List[ModelThresholdReport] = []
    baseline_thr = float(args.baseline_threshold)
    grid_size = int(args.grid_size)

    print("=" * 90)
    print("OOF THRESHOLD TUNING (maximize F1)")
    print(f"Stage1 source: {args.data_stage1}")
    print(f"Label col:     {label_col}")
    print(f"Rows in df:    {len(df)}")
    print(f"Models:        {', '.join(model_paths.keys())}")
    print(f"Sweep grid:    {grid_size} thresholds from 0 to 1")
    print(f"Baseline thr:  {baseline_thr}")
    print("=" * 90)

    summary_rows_csv: List[Dict[str, object]] = []

    for model_name, npy_path in model_paths.items():
        try:
            y_score = _load_npy_1d(npy_path)
        except Exception as e:
            print(f"[ERROR] Failed loading npy for {model_name}: {e}", file=sys.stderr)
            sys.exit(1)

        if y_score.shape[0] != y.shape[0]:
            print(f"[ERROR] Length mismatch for {model_name}.", file=sys.stderr)
            print(f"  len(df/y)={y.shape[0]}  len(npy)={y_score.shape[0]}", file=sys.stderr)
            print("  This usually means the .npy was generated from a different dataset/order.", file=sys.stderr)
            sys.exit(1)

        mask = _finite_mask(y_score.astype(float), y.astype(float))
        y_used = y[mask]
        s_used = y_score[mask]

        pos_rate = float(np.mean(y_used)) if y_used.size else float("nan")

        pr_auc = _safe_auc(average_precision_score, y_used, s_used)
        roc_auc = _safe_auc(roc_auc_score, y_used, s_used)

        m05 = _metrics_at_threshold(y_used, s_used, baseline_thr)
        best_thr, mbest = _sweep_best_f1(y_used, s_used, grid_size=grid_size)

        rep = ModelThresholdReport(
            model=model_name,
            npy_path=str(npy_path),
            n_rows_df=int(len(df)),
            n_rows_used=int(y_used.size),
            pos_rate_used=pos_rate,
            pr_auc=pr_auc,
            roc_auc=roc_auc,
            metrics_t05=m05,
            best_threshold=float(best_thr),
            metrics_best=mbest,
        )
        reports.append(rep)

        # console report
        print("\n" + "-" * 90)
        print(f"MODEL: {model_name}")
        print(f"OOF npy: {npy_path}")
        print(f"Used rows (finite preds): {rep.n_rows_used}/{rep.n_rows_df} ({rep.n_rows_used/rep.n_rows_df:.2%})")
        print(f"Positive rate (used):     {rep.pos_rate_used:.4f}")
        print(f"PR-AUC:                  {rep.pr_auc if rep.pr_auc is not None else 'N/A'}")
        print(f"ROC-AUC:                 {rep.roc_auc if rep.roc_auc is not None else 'N/A'}")

        print("\nBEFORE (baseline threshold):")
        print(f"  t= {baseline_thr:.3f} | F1={m05['f1']:.4f} | P={m05['precision']:.4f} | R={m05['recall']:.4f} | Acc={m05['accuracy']:.4f}")

        print("\nAFTER (best F1 threshold):")
        print(f"  t*= {best_thr:.3f} | F1={mbest['f1']:.4f} | P={mbest['precision']:.4f} | R={mbest['recall']:.4f} | Acc={mbest['accuracy']:.4f}")

        # row for CSV
        summary_rows_csv.append(
            {
                "model": model_name,
                "n_rows_df": rep.n_rows_df,
                "n_rows_used": rep.n_rows_used,
                "pos_rate_used": rep.pos_rate_used,
                "pr_auc": rep.pr_auc,
                "roc_auc": rep.roc_auc,
                "baseline_threshold": baseline_thr,
                "baseline_f1": m05["f1"],
                "baseline_precision": m05["precision"],
                "baseline_recall": m05["recall"],
                "baseline_accuracy": m05["accuracy"],
                "best_threshold": best_thr,
                "best_f1": mbest["f1"],
                "best_precision": mbest["precision"],
                "best_recall": mbest["recall"],
                "best_accuracy": mbest["accuracy"],
            }
        )

    # save outputs
    summary_csv_path = out_dir / "threshold_tuning_summary.csv"
    _write_csv(summary_rows_csv, summary_csv_path)

    json_path = out_dir / "threshold_tuning_summary.json"
    with json_path.open("w") as f:
        json.dump([asdict(r) for r in reports], f, indent=2)

    print("\n" + "=" * 90)
    print(f"Saved CSV:  {summary_csv_path}")
    print(f"Saved JSON: {json_path}")
    print("=" * 90)

    # also print a quick "recommended thresholds" line
    print("\nRecommended thresholds (t* maximizing F1):")
    for r in reports:
        print(f"  {r.model}: {r.best_threshold:.3f} (F1={r.metrics_best['f1']:.4f}, P={r.metrics_best['precision']:.4f}, R={r.metrics_best['recall']:.4f})")


if __name__ == "__main__":
    main()
