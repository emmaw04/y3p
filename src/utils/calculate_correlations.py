# src/utils/calculate_correlations.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _load_stage1_df(data_stage1_path: str):
    """
    Load stage1 dataframe using the project loader so filtering (e.g. HOLDOUT_RACE_IDS)
    matches evaluate_base.py.
    """
    try:
        from src.data.data import load_stage1_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import src.data.data.load_stage1_dataset. "
            "Run this from repo root with: python -m src.utils.calculate_correlations ..."
        ) from e

    # Be robust to different function signatures.
    try:
        return load_stage1_dataset(data_stage1_path)
    except TypeError:
        # some versions use strict=...
        try:
            return load_stage1_dataset(data_stage1_path, strict=True)
        except TypeError:
            return load_stage1_dataset(data_stage1_path, strict=False)


def _load_npy(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing .npy file: {p}")
    arr = np.load(p)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(float)


def _valid_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # drop NaNs / infs (common for sequence OOF arrays)
    return np.isfinite(a) & np.isfinite(b)


def _corr_safe(func, x: np.ndarray, y: np.ndarray) -> Tuple[float, Optional[float]]:
    """
    Compute correlation safely.
    Returns (corr, pvalue). corr is np.nan if not computable.
    """
    if x.size < 2:
        return (float("nan"), None)
    # If one side is constant, scipy raises warnings / returns nan; handle explicitly
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return (float("nan"), None)
    c, p = func(x, y)
    return (float(c), float(p))


def _disagreement(a: np.ndarray, b: np.ndarray, thr: float) -> Tuple[int, int, int, float]:
    """
    Returns:
      a_not_b, b_not_a, total, rate
    computed only on provided arrays (assumed aligned & already filtered to valid indices).
    """
    a_pit = a >= thr
    b_pit = b >= thr
    a_not_b = int(np.sum(a_pit & ~b_pit))
    b_not_a = int(np.sum(b_pit & ~a_pit))
    total = a_not_b + b_not_a
    rate = float(total / a.size) if a.size > 0 else float("nan")
    return a_not_b, b_not_a, total, rate


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare two OOF probability arrays (global + by race) with holdout filtering via load_stage1_dataset()."
    )
    ap.add_argument(
        "--a",
        required=True,
        help="Path to Model A oof_pred_proba_pos.npy",
    )
    ap.add_argument(
        "--b",
        required=True,
        help="Path to Model B oof_pred_proba_pos.npy",
    )
    ap.add_argument(
        "--data_stage1",
        required=True,
        help="Path to the stage1 CSV used by evaluate_base.py (will be loaded via load_stage1_dataset to remove holdouts).",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used for disagreement rate (default: 0.5).",
    )
    ap.add_argument(
        "--max_races",
        type=int,
        default=0,
        help="If >0, only print the first N races (useful if output is huge).",
    )
    args = ap.parse_args()

    # --- load arrays ---
    try:
        preds_a = _load_npy(args.a)
        preds_b = _load_npy(args.b)
    except Exception as e:
        print(f"[ERROR] Failed loading npy files: {e}", file=sys.stderr)
        sys.exit(1)

    # --- load df (filters holdouts inside load_stage1_dataset) ---
    try:
        df = _load_stage1_df(args.data_stage1)
    except Exception as e:
        print(f"[ERROR] Failed loading stage1 dataset via load_stage1_dataset: {e}", file=sys.stderr)
        sys.exit(1)

    if "race_id" not in df.columns:
        print("[ERROR] stage1 dataframe does not contain 'race_id' column.", file=sys.stderr)
        sys.exit(1)

    race_ids = df["race_id"].to_numpy()

    # --- alignment checks ---
    if preds_a.shape[0] != preds_b.shape[0]:
        print("[ERROR] preds_a and preds_b lengths differ.", file=sys.stderr)
        print(f"  len(a)={preds_a.shape[0]} len(b)={preds_b.shape[0]}", file=sys.stderr)
        sys.exit(1)

    if preds_a.shape[0] != race_ids.shape[0]:
        print("[ERROR] .npy length does not match filtered stage1 dataframe length.", file=sys.stderr)
        print(f"  len(npy)={preds_a.shape[0]} len(df_after_load_stage1_dataset)={race_ids.shape[0]}", file=sys.stderr)
        print("  This usually means the .npy was generated from a different dataset/version.", file=sys.stderr)
        sys.exit(1)

    thr = float(args.threshold)

    # --- global metrics (drop NaNs/infs first) ---
    valid = _valid_mask(preds_a, preds_b)
    a = preds_a[valid]
    b = preds_b[valid]
    races_valid = race_ids[valid]

    print("=" * 80)
    print("OOF CORRELATION REPORT (with holdout filtering via load_stage1_dataset)")
    print(f"Model A: {args.a}")
    print(f"Model B: {args.b}")
    print(f"Stage1 CSV (loaded+filtered): {args.data_stage1}")
    print(f"Threshold: {thr}")
    print(f"Total rows in df: {len(df)}")
    print(f"Valid comparable rows (finite in both A & B): {a.size} ({a.size/len(df):.2%})")
    print("=" * 80)

    pear, pear_p = _corr_safe(pearsonr, a, b)
    spear, spear_p = _corr_safe(spearmanr, a, b)
    a_not_b, b_not_a, total_dis, dis_rate = _disagreement(a, b, thr)

    print("\n--- Global metrics (finite rows only) ---")
    if np.isfinite(pear):
        print(f"Pearson r:  {pear:.4f}" + (f"  (p={pear_p:.3g})" if pear_p is not None else ""))
    else:
        print("Pearson r:  nan (not enough variation or <2 points)")

    if np.isfinite(spear):
        print(f"Spearman ρ: {spear:.4f}" + (f"  (p={spear_p:.3g})" if spear_p is not None else ""))
    else:
        print("Spearman ρ: nan (not enough variation or <2 points)")

    print(f"Disagreement rate @ {thr:.2f}: {dis_rate:.4f}")
    print(f"  A predicts pit & B doesn't: {a_not_b} laps")
    print(f"  B predicts pit & A doesn't: {b_not_a} laps")
    print(f"  Total disagreements:        {total_dis} laps (out of {a.size})")

    # --- per-race metrics ---
    print("\n--- By-race metrics (finite rows only) ---")
    unique_races = np.unique(races_valid)

    shown = 0
    for rid in unique_races:
        idx = np.where(races_valid == rid)[0]
        if idx.size < 2:
            continue

        ar = a[idx]
        br = b[idx]

        pr, _ = _corr_safe(pearsonr, ar, br)
        sr, _ = _corr_safe(spearmanr, ar, br)
        a_nb, b_na, tdis, dr = _disagreement(ar, br, thr)

        print(f"Race {rid}: n={idx.size:5d} | Pearson={pr: .4f} | Spearman={sr: .4f} | disagree={dr:.4f} ({tdis}/{idx.size})")

        shown += 1
        if args.max_races and shown >= args.max_races:
            print(f"... stopped after {args.max_races} races (set --max_races 0 to print all).")
            break


if __name__ == "__main__":
    main()
