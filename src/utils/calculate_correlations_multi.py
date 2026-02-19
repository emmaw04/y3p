# src/utils/calculate_correlations_multi.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
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
            "Run this from repo root with: python -m src.utils.calculate_correlations_multi ..."
        ) from e

    # Be robust to different function signatures.
    try:
        return load_stage1_dataset(data_stage1_path)
    except TypeError:
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


def _corr_safe(func, x: np.ndarray, y: np.ndarray) -> Tuple[float, Optional[float]]:
    """
    Compute correlation safely.
    Returns (corr, pvalue). corr is np.nan if not computable.
    """
    if x.size < 2:
        return (float("nan"), None)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return (float("nan"), None)
    c, p = func(x, y)
    return (float(c), float(p))


def _disagreement(a: np.ndarray, b: np.ndarray, thr: float) -> Tuple[int, int, int, float]:
    a_pit = a >= thr
    b_pit = b >= thr
    a_not_b = int(np.sum(a_pit & ~b_pit))
    b_not_a = int(np.sum(b_pit & ~a_pit))
    total = a_not_b + b_not_a
    rate = float(total / a.size) if a.size > 0 else float("nan")
    return a_not_b, b_not_a, total, rate


def _parse_model_arg(s: str) -> Tuple[str, str]:
    """
    Parse --model NAME=PATH (or NAME:PATH).
    """
    if "=" in s:
        name, path = s.split("=", 1)
    elif ":" in s:
        name, path = s.split(":", 1)
    else:
        raise ValueError("Model spec must be NAME=PATH (or NAME:PATH).")
    name = name.strip()
    path = path.strip()
    if not name:
        raise ValueError("Empty model name in --model.")
    if not path:
        raise ValueError("Empty path in --model.")
    return name, path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute pairwise Pearson/Spearman correlations and disagreement rates "
            "between multiple OOF probability arrays, globally + by-race, with holdout "
            "filtering via load_stage1_dataset()."
        )
    )
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec as NAME=PATH (repeat for multiple models). Example: --model tcn=.../oof.npy",
    )
    ap.add_argument(
        "--data_stage1",
        required=True,
        help="Path to stage1 CSV used by evaluate_base.py (loaded via load_stage1_dataset to remove holdouts).",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for disagreement rate (default: 0.5).",
    )
    ap.add_argument(
        "--by_race",
        action="store_true",
        help="If set, compute by-race pairwise metrics and save to CSV (can be large).",
    )
    ap.add_argument(
        "--max_races",
        type=int,
        default=0,
        help="If >0 and --by_race, only process the first N races (for quick checks).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Optional directory to save CSV outputs. If empty, saves nothing.",
    )
    args = ap.parse_args()

    # Parse model specs
    try:
        specs = [_parse_model_arg(s) for s in (args.model or [])]
    except Exception as e:
        print(f"[ERROR] Bad --model argument: {e}", file=sys.stderr)
        sys.exit(1)

    if len(specs) < 2:
        print("[ERROR] Provide at least 2 models via --model.", file=sys.stderr)
        sys.exit(1)

    names = [n for n, _ in specs]
    if len(set(names)) != len(names):
        print("[ERROR] Duplicate model names in --model.", file=sys.stderr)
        sys.exit(1)

    # Load stage1 df (filters holdouts)
    try:
        df = _load_stage1_df(args.data_stage1)
    except Exception as e:
        print(f"[ERROR] Failed loading stage1 dataset via load_stage1_dataset: {e}", file=sys.stderr)
        sys.exit(1)

    if "race_id" not in df.columns:
        print("[ERROR] stage1 dataframe does not contain 'race_id' column.", file=sys.stderr)
        sys.exit(1)

    race_ids = df["race_id"].to_numpy()
    n_rows = race_ids.shape[0]

    # Load all predictions
    preds: Dict[str, np.ndarray] = {}
    for name, path in specs:
        try:
            arr = _load_npy(path)
        except Exception as e:
            print(f"[ERROR] Failed loading {name} npy: {e}", file=sys.stderr)
            sys.exit(1)

        if arr.shape[0] != n_rows:
            print("[ERROR] .npy length does not match filtered stage1 dataframe length.", file=sys.stderr)
            print(f"  model={name} len(npy)={arr.shape[0]} len(df_after_load_stage1_dataset)={n_rows}", file=sys.stderr)
            print("  This usually means the .npy was generated from a different dataset/version.", file=sys.stderr)
            sys.exit(1)

        preds[name] = arr

    thr = float(args.threshold)

    # Prepare output dir
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("OOF MULTI-MODEL CORRELATION REPORT (with holdout filtering via load_stage1_dataset)")
    print(f"Stage1 CSV (loaded+filtered): {args.data_stage1}")
    print(f"Models: {', '.join(names)}")
    print(f"Threshold: {thr}")
    print(f"Rows in df: {n_rows}")
    print("=" * 90)

    # -------- Global pairwise summary (long-form) --------
    global_rows: List[dict] = []
    for a_name, b_name in combinations(names, 2):
        a_raw = preds[a_name]
        b_raw = preds[b_name]
        valid = np.isfinite(a_raw) & np.isfinite(b_raw)
        a = a_raw[valid]
        b = b_raw[valid]

        pear, pear_p = _corr_safe(pearsonr, a, b)
        spear, spear_p = _corr_safe(spearmanr, a, b)
        a_not_b, b_not_a, tdis, dr = _disagreement(a, b, thr)

        global_rows.append(
            dict(
                model_a=a_name,
                model_b=b_name,
                n_valid=int(a.size),
                valid_frac=float(a.size / n_rows) if n_rows else float("nan"),
                pearson=pear,
                pearson_p=pear_p if pear_p is not None else float("nan"),
                spearman=spear,
                spearman_p=spear_p if spear_p is not None else float("nan"),
                disagree_rate=dr,
                a_not_b=a_not_b,
                b_not_a=b_not_a,
                total_disagree=tdis,
            )
        )

    gdf = pd.DataFrame(global_rows).sort_values(["disagree_rate", "pearson"], ascending=[False, True])

    print("\n--- Global pairwise metrics (finite rows only; pairwise mask) ---")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 160):
        print(gdf.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}"))

    if out_dir:
        g_path = out_dir / "global_pairwise_metrics.csv"
        gdf.to_csv(g_path, index=False)
        print(f"\nSaved: {g_path}")

    # -------- By-race (optional) --------
    if args.by_race:
        print("\n--- By-race metrics ---")
        unique_races = np.unique(race_ids)
        if args.max_races and args.max_races > 0:
            unique_races = unique_races[: args.max_races]

        by_race_rows: List[dict] = []

        # Precompute indices per race once
        race_to_idx = {rid: np.where(race_ids == rid)[0] for rid in unique_races}

        for rid in unique_races:
            idx = race_to_idx[rid]
            if idx.size < 2:
                continue

            for a_name, b_name in combinations(names, 2):
                a_raw = preds[a_name][idx]
                b_raw = preds[b_name][idx]
                valid = np.isfinite(a_raw) & np.isfinite(b_raw)
                a = a_raw[valid]
                b = b_raw[valid]
                if a.size < 2:
                    continue

                pear, _ = _corr_safe(pearsonr, a, b)
                spear, _ = _corr_safe(spearmanr, a, b)
                a_not_b, b_not_a, tdis, dr = _disagreement(a, b, thr)

                by_race_rows.append(
                    dict(
                        race_id=int(rid),
                        model_a=a_name,
                        model_b=b_name,
                        n_valid=int(a.size),
                        pearson=pear,
                        spearman=spear,
                        disagree_rate=dr,
                        total_disagree=tdis,
                        a_not_b=a_not_b,
                        b_not_a=b_not_a,
                    )
                )

        rdf = pd.DataFrame(by_race_rows)
        if rdf.empty:
            print("No by-race rows produced (likely too many NaNs or very short races).")
        else:
            # Print a compact “worst disagreement” snapshot
            worst = rdf.sort_values("disagree_rate", ascending=False).head(20)
            print("\nTop 20 race/pair by disagreement rate:")
            with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 160):
                print(worst.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}"))

            if out_dir:
                r_path = out_dir / "by_race_pairwise_metrics.csv"
                rdf.to_csv(r_path, index=False)
                print(f"\nSaved: {r_path}")
            else:
                print("\nTip: set --out_dir to save full by-race CSV (recommended).")


if __name__ == "__main__":
    main()

'''
python -m src.utils.calculate_correlations_multi \
  --data_stage1 data/stage1.csv \
  --threshold 0.5 \
  --by_race \
  --out_dir results/runs/seq/stage1/binary/ensemble_diagnostics \
  --model gru=results/runs/seq/stage1/binary/gru/oof_pred_proba_pos.npy \
  --model lstm=results/runs/seq/stage1/binary/lstm/oof_pred_proba_pos.npy \
  --model tcn=results/runs/seq/stage1/binary/tcn/oof_pred_proba_pos.npy \
  --model tcn_gru=results/runs/seq/stage1/binary/tcn_gru/oof_pred_proba_pos.npy

'''