#!/usr/bin/env python3
# src/utils/SHAP_eval_b.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from joblib import load

from sklearn.metrics import average_precision_score, log_loss, f1_score, precision_score, recall_score

# project imports
from src.data.data import HOLDOUT_RACE_IDS, build_feature_sequences

# keras for ANN + TCN saved models
import tensorflow as tf

# SHAP (TreeSHAP)
try:
    import shap  # type: ignore
except Exception as e:
    shap = None


# -----------------------------
# small utils
# -----------------------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def vprint(verbose: bool, *args, **kwargs) -> None:
    if verbose:
        print(*args, **kwargs, flush=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_predict_keras_binary(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Returns p(y=1) as shape (n,).
    """
    out = model.predict(X, verbose=0)
    out = np.asarray(out)
    if out.ndim == 2 and out.shape[1] == 1:
        return out[:, 0].astype(float)
    if out.ndim == 1:
        return out.astype(float)
    # if model outputs logits or 2-unit softmax unexpectedly
    if out.ndim == 2 and out.shape[1] == 2:
        return out[:, 1].astype(float)
    raise ValueError(f"Unexpected keras output shape: {out.shape}")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def metrics_binary(y_true: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (proba >= thr).astype(int)
    return {
        "PR_AUC": float(average_precision_score(y_true, proba)),
        "LogLoss": float(log_loss(y_true, proba, labels=[0, 1])),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        **{k: float(v) for k, v in confusion_counts(y_true, y_pred).items()},
    }


# -----------------------------
# feature / grouping setup
# -----------------------------
BASE_PROB_FEATURES = ["p_svm", "p_rf", "p_xgb", "p_ann", "tcn_proba", "tcn_effective_len"]

RAW_GROUPS: Dict[str, List[str]] = {
    "G_race_phase": ["lapno", "race_progress"],
    "G_race_control": ["fcy_status"],
    "G_track": ["race_track", "track_category"],
    "G_tyre": ["current_compound", "tyre_age", "fulfilled_second_compound"],
    "G_traffic": ["position", "interval", "close_ahead"],
    "G_strategy_history": ["pit_stops_so_far", "tyre_change_pursuer"],
    "G_pace": ["lap_time"],
    "G_weather": ["rained_yet", "is_raining", "minutes_rain"],
}


def feature_group(feature_name: str) -> str:
    if feature_name in BASE_PROB_FEATURES:
        return "BASE_SIGNALS"
    for g, cols in RAW_GROUPS.items():
        if feature_name in cols:
            return g
    return "OTHER"


# -----------------------------
# pipeline introspection
# -----------------------------
def split_pipeline(pipe) -> Tuple[Any, Any]:
    """
    Return (preprocessor, model) from a sklearn Pipeline-ish object.
    Assumes last step is model and an earlier step is preprocessor.
    """
    if hasattr(pipe, "named_steps"):
        ns = pipe.named_steps
        if "model" in ns:
            model = ns["model"]
            # pick first non-model as pre
            pre = None
            for k, v in ns.items():
                if k != "model":
                    pre = v
                    break
            if pre is None:
                raise RuntimeError("Pipeline has 'model' but no preprocessor step.")
            return pre, model
        # fallback: last step model, first step pre
        steps = list(ns.values())
        if len(steps) >= 2:
            return steps[0], steps[-1]
    # no named_steps -> treat as model only
    return None, pipe


def get_feature_names_out(pre, X: pd.DataFrame) -> List[str]:
    """
    Try to obtain transformed feature names from sklearn preprocessor/ColumnTransformer.
    """
    if pre is None:
        return list(X.columns)

    # sklearn >=1.0: many transformers implement get_feature_names_out
    try:
        names = pre.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        pass

    # fallback: try with input features
    try:
        names = pre.get_feature_names_out(X.columns)
        return [str(n) for n in names]
    except Exception:
        pass

    # last resort
    return [f"f{i}" for i in range(getattr(pre, "n_features_in_", X.shape[1]))]


def map_transformed_to_original(
    transformed_names: List[str],
    original_numeric: List[str],
    original_cats: List[str],
) -> List[str]:
    """
    Map a transformed feature name (often like 'cat__current_compound_SOFT')
    back to its original feature (like 'current_compound').

    Uses a robust prefix match against known categorical columns.
    """
    cat_cols_sorted = sorted(original_cats, key=len, reverse=True)
    out: List[str] = []
    for t in transformed_names:
        # strip transformer prefix e.g. "num__" / "cat__" / "remainder__"
        base = t.split("__", 1)[1] if "__" in t else t

        # exact numeric match
        if base in original_numeric:
            out.append(base)
            continue

        # categorical one-hot: match the longest cat col prefix
        matched = None
        for c in cat_cols_sorted:
            if base == c:
                matched = c
                break
            if base.startswith(c + "_") or base.startswith(c + "="):
                matched = c
                break
        if matched is not None:
            out.append(matched)
        else:
            # unknown remainder; keep as-is
            out.append(base)
    return out


# -----------------------------
# build meta inputs on holdouts
# -----------------------------
def load_holdout_df(data_stage1: Path, verbose: bool) -> pd.DataFrame:
    df = pd.read_csv(data_stage1)

    # normalize types (race_id often int)
    if "race_id" not in df.columns:
        raise ValueError("stage1 CSV missing required column: race_id")
    df["race_id"] = df["race_id"].astype(int)

    holdouts = set(int(x) for x in HOLDOUT_RACE_IDS)
    df_h = df[df["race_id"].isin(holdouts)].copy()

    if len(df_h) == 0:
        raise ValueError(
            f"No rows found for HOLDOUT_RACE_IDS={sorted(list(holdouts))} in {data_stage1}."
        )

    # require label
    if "y_pit" not in df_h.columns:
        raise ValueError("stage1 CSV missing required label column: y_pit")

    df_h["y_pit"] = df_h["y_pit"].astype(int)
    vprint(verbose, f"[data] holdout rows={len(df_h)} races={sorted(df_h['race_id'].unique().tolist())}")
    return df_h


def build_base_feature_frame(df_h: pd.DataFrame, X1_cols: List[str]) -> pd.DataFrame:
    # Use saved X1_columns to match training time feature set/order
    missing = [c for c in X1_cols if c not in df_h.columns]
    if missing:
        raise ValueError(f"Holdout df missing expected X1 columns from artifacts: {missing[:10]}{'...' if len(missing)>10 else ''}")

    X = df_h[X1_cols].copy()
    # hard-drop leakage columns if they accidentally appear
    X = X.drop(columns=["y_pit", "row_id", "race_id", "driver_id"], errors="ignore")
    return X


def predict_base_probs(
    artifacts_dir: Path,
    X_base: pd.DataFrame,
    verbose: bool,
) -> Dict[str, np.ndarray]:
    """
    Produces p_svm, p_rf, p_xgb (sklearn pipelines) and p_ann (keras saved).
    """
    out: Dict[str, np.ndarray] = {}

    # sklearn pipelines
    for name in ["svm", "rf", "xgb"]:
        p = artifacts_dir / f"base_{name}_pipeline.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Missing base pipeline: {p}")
        pipe = load(p)
        proba = pipe.predict_proba(X_base)[:, 1].astype(float)
        out[f"p_{name}"] = proba
        vprint(verbose, f"[base] {name}: mean p={float(np.mean(proba)):.4f}")

    # ANN saved as preprocessor + keras model
    ann_pre_path = artifacts_dir / "ann_preprocessor.joblib"
    ann_model_path = artifacts_dir / "ann_model.keras"
    if not ann_pre_path.exists() or not ann_model_path.exists():
        raise FileNotFoundError(f"Missing ANN artifacts: {ann_pre_path} or {ann_model_path}")

    ann_pre = load(ann_pre_path)
    Xt = ann_pre.transform(X_base)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt).astype(np.float32)

    ann_model = tf.keras.models.load_model(ann_model_path)
    p_ann = safe_predict_keras_binary(ann_model, Xt)
    out["p_ann"] = p_ann
    vprint(verbose, f"[base] ann: mean p={float(np.mean(p_ann)):.4f}")

    return out


def predict_tcn_features(
    artifacts_dir: Path,
    df_h: pd.DataFrame,
    tcn_cfg: Dict[str, Any],
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      tcn_proba (N,)
      tcn_effective_len_scaled (N,) in [0,1]
    """
    seq_len = int(tcn_cfg.get("seq_len", 8))
    pad_left = bool(tcn_cfg.get("pad_left", True))
    add_timestep_mask = bool(tcn_cfg.get("add_timestep_mask", True))

    pre_path = artifacts_dir / "tcn_preprocessor.joblib"
    model_path = artifacts_dir / "tcn_model.keras"
    if not pre_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing TCN artifacts: {pre_path} or {model_path}")

    # IMPORTANT: drop ID + label + row_id (no leakage)
    X_tab = df_h.drop(
        columns=["y_pit", "y_compound", "race_id", "driver_id", "row_id"],
        errors="ignore",
    ).copy()

    y_full = df_h["y_pit"].astype(int)
    keys = df_h[["race_id", "driver_id", "lapno"]].copy()

    tcn_pre = load(pre_path)
    Xt_all = tcn_pre.transform(X_tab)
    if hasattr(Xt_all, "toarray"):
        Xt_all = Xt_all.toarray()
    Xt_all = np.asarray(Xt_all).astype(np.float32)

    X_seq, y_seq, idx_last, seq_idx, eff_len = build_feature_sequences(
        keys=keys,
        Xt_all=Xt_all,
        y_full=y_full.to_numpy(),
        seq_len=seq_len,
        pad_left=pad_left,
        add_timestep_mask=add_timestep_mask,
    )

    tcn_model = tf.keras.models.load_model(model_path)
    proba_seq = safe_predict_keras_binary(tcn_model, X_seq.astype(np.float32))

    N = len(df_h)
    tcn_proba = np.full(N, np.nan, dtype=float)
    tcn_eff = np.zeros(N, dtype=float)
    tcn_proba[idx_last] = proba_seq
    tcn_eff[idx_last] = eff_len.astype(float)

    tcn_proba = np.nan_to_num(tcn_proba, nan=0.0)
    tcn_eff_scaled = tcn_eff / float(seq_len)

    vprint(verbose, f"[tcn] seq_len={seq_len} pad_left={pad_left} tmask={add_timestep_mask} mean p={float(np.mean(tcn_proba)):.4f}")
    return tcn_proba, tcn_eff_scaled


def build_meta_df(
    df_h: pd.DataFrame,
    meta_cols: List[str],
    base_probs: Dict[str, np.ndarray],
    tcn_proba: np.ndarray,
    tcn_eff_scaled: np.ndarray,
) -> pd.DataFrame:
    meta = pd.DataFrame(index=df_h.index)

    # inject probability features
    for k, v in base_probs.items():
        meta[k] = v
    meta["tcn_proba"] = tcn_proba
    meta["tcn_effective_len"] = tcn_eff_scaled

    # inject raw/context features that the meta model expects
    for col in meta_cols:
        if col in meta.columns:
            continue
        if col in df_h.columns:
            meta[col] = df_h[col]
        else:
            raise ValueError(f"Meta feature '{col}' not found in base probs or holdout dataframe columns.")

    # enforce exact order
    meta = meta[meta_cols].copy()
    return meta


# -----------------------------
# cohort selection
# -----------------------------
def assign_cohort(y_true: np.ndarray, proba: np.ndarray, thr: float) -> np.ndarray:
    y_pred = (proba >= thr).astype(int)
    cohort = np.full(len(y_true), "TN", dtype=object)
    cohort[(y_true == 1) & (y_pred == 1)] = "TP"
    cohort[(y_true == 0) & (y_pred == 1)] = "FP"
    cohort[(y_true == 1) & (y_pred == 0)] = "FN"
    return cohort


def select_case_studies(
    df: pd.DataFrame,
    per_race: int,
    thr: float,
) -> pd.DataFrame:
    """
    For each race:
      - top N TP by proba
      - top N FP by proba
      - top N FN closest to threshold (near misses)
    """
    out_parts: List[pd.DataFrame] = []
    for race_id, g in df.groupby("race_id", sort=True):
        tp = g[g["cohort"] == "TP"].nlargest(per_race, "proba")
        fp = g[g["cohort"] == "FP"].nlargest(per_race, "proba")
        fn = g[g["cohort"] == "FN"].copy()
        if len(fn) > 0:
            fn["dist_to_thr"] = (thr - fn["proba"]).abs()
            fn = fn.nsmallest(per_race, "dist_to_thr")
            fn = fn.drop(columns=["dist_to_thr"])
        out_parts.append(tp)
        out_parts.append(fp)
        out_parts.append(fn)
    return pd.concat(out_parts, axis=0).drop_duplicates(subset=["row_id"], keep="first")


# -----------------------------
# SHAP computation + aggregation
# -----------------------------
def compute_shap_tree(
    meta_pipe,
    X_meta: pd.DataFrame,
    verbose: bool,
):
    if shap is None:
        raise RuntimeError(
            "shap is not installed in this environment. Install with: pip install shap"
        )

    pre, model = split_pipeline(meta_pipe)
    Xt = pre.transform(X_meta) if pre is not None else X_meta.to_numpy()

    # convert sparse -> dense for shap stability on small-ish heldout sets
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xt)

    # binary: shap may return list [class0, class1] or (n,m)
    if isinstance(sv, list):
        # prefer class 1 contributions
        shap_vals = np.asarray(sv[1])
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        shap_vals = np.asarray(sv)
        expected_value = explainer.expected_value

    vprint(verbose, f"[shap] computed shap_vals shape={shap_vals.shape}")
    return shap_vals, float(expected_value), pre, model, Xt


def aggregate_shap_to_original(
    shap_vals: np.ndarray,
    transformed_names: List[str],
    X_meta: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      shap_signed_orig: (n, n_orig_features) signed sums
      shap_abs_orig:    (n, n_orig_features) abs sums
    """
    num_cols = [c for c in X_meta.columns if pd.api.types.is_numeric_dtype(X_meta[c])]
    cat_cols = [c for c in X_meta.columns if not pd.api.types.is_numeric_dtype(X_meta[c])]

    orig_map = map_transformed_to_original(transformed_names, num_cols, cat_cols)

    # build index lists per original feature
    by_feat: Dict[str, List[int]] = {}
    for j, orig in enumerate(orig_map):
        by_feat.setdefault(orig, []).append(j)

    feats = list(by_feat.keys())
    signed = np.zeros((shap_vals.shape[0], len(feats)), dtype=float)
    absv = np.zeros((shap_vals.shape[0], len(feats)), dtype=float)

    for i, f in enumerate(feats):
        idxs = by_feat[f]
        sv = shap_vals[:, idxs]
        signed[:, i] = np.sum(sv, axis=1)
        absv[:, i] = np.sum(np.abs(sv), axis=1)

    shap_signed_orig = pd.DataFrame(signed, columns=feats)
    shap_abs_orig = pd.DataFrame(absv, columns=feats)
    return shap_signed_orig, shap_abs_orig


def top_contributors_long(
    df_info: pd.DataFrame,
    shap_signed_orig: pd.DataFrame,
    shap_abs_orig: pd.DataFrame,
    X_meta: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """
    Long-format top contributors per instance (aggregated to original features).
    """
    rows: List[Dict[str, Any]] = []
    feats = list(shap_abs_orig.columns)

    for i in range(len(df_info)):
        abs_row = shap_abs_orig.iloc[i].to_numpy()
        signed_row = shap_signed_orig.iloc[i].to_numpy()

        top_idx = np.argsort(-abs_row)[:top_k]
        for rank, j in enumerate(top_idx, start=1):
            feat = feats[int(j)]
            rows.append({
                "race_id": int(df_info.iloc[i]["race_id"]),
                "driver_id": int(df_info.iloc[i]["driver_id"]) if "driver_id" in df_info.columns else np.nan,
                "lapno": int(df_info.iloc[i]["lapno"]),
                "row_id": int(df_info.iloc[i]["row_id"]) if "row_id" in df_info.columns else np.nan,
                "y_true": int(df_info.iloc[i]["y_true"]),
                "proba": float(df_info.iloc[i]["proba"]),
                "cohort": str(df_info.iloc[i]["cohort"]),
                "rank": int(rank),
                "feature": feat,
                "shap_signed": float(signed_row[int(j)]),
                "shap_abs": float(abs_row[int(j)]),
                "feature_value": X_meta.iloc[i][feat] if feat in X_meta.columns else np.nan,
                "group": feature_group(feat),
                "direction": "push_PIT" if signed_row[int(j)] > 0 else "push_NO_PIT",
            })
    return pd.DataFrame(rows)


def cohort_importance(
    df_info: pd.DataFrame,
    shap_abs_orig: pd.DataFrame,
) -> pd.DataFrame:
    """
    mean(|SHAP|) per feature per cohort
    """
    out_parts: List[pd.DataFrame] = []
    for cohort, idx in df_info.groupby("cohort").groups.items():
        abs_mean = shap_abs_orig.iloc[list(idx)].mean(axis=0).sort_values(ascending=False)
        tmp = abs_mean.reset_index()
        tmp.columns = ["feature", "mean_abs_shap"]
        tmp["cohort"] = cohort
        tmp["rank"] = np.arange(1, len(tmp) + 1)
        tmp["group"] = tmp["feature"].map(feature_group)
        out_parts.append(tmp)
    return pd.concat(out_parts, axis=0, ignore_index=True)


def grouped_attribution(
    df_info: pd.DataFrame,
    shap_abs_orig: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each instance: base vs groups share, then aggregate by cohort (+ race).
    """
    feats = list(shap_abs_orig.columns)
    feat_to_group = {f: feature_group(f) for f in feats}

    # per-instance sums
    group_names = sorted(set(feat_to_group.values()))
    mat = shap_abs_orig.to_numpy()
    total = np.sum(mat, axis=1) + 1e-12

    group_sums = {g: np.zeros(len(df_info), dtype=float) for g in group_names}
    for j, f in enumerate(feats):
        group_sums[feat_to_group[f]] += mat[:, j]

    df_inst = df_info[["race_id", "cohort"]].copy()
    for g in group_names:
        df_inst[f"share_{g}"] = group_sums[g] / total

    # aggregate by race + cohort
    agg = df_inst.groupby(["race_id", "cohort"]).mean(numeric_only=True).reset_index()
    return agg


# -----------------------------
# plotting (simple + thesis-friendly)
# -----------------------------
def plot_cohort_bar(
    outpath: Path,
    imp: pd.DataFrame,
    cohort: str,
    top_n: int = 20,
):
    import matplotlib.pyplot as plt

    d = imp[imp["cohort"] == cohort].sort_values("mean_abs_shap", ascending=False).head(top_n)
    if len(d) == 0:
        return

    plt.figure(figsize=(10, 6))
    plt.barh(d["feature"][::-1], d["mean_abs_shap"][::-1])
    plt.title(f"Meta SHAP mean(|SHAP|) — {cohort} (top {top_n})")
    plt.xlabel("mean(|SHAP|) (log-odds units)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_instance_contrib(
    outpath: Path,
    df_row: pd.Series,
    contribs: pd.DataFrame,
    top_n: int = 10,
):
    import matplotlib.pyplot as plt

    # contribs already filtered for this instance
    d = contribs.sort_values("shap_abs", ascending=False).head(top_n).copy()
    if len(d) == 0:
        return

    d = d.sort_values("shap_signed")  # negatives bottom, positives top
    plt.figure(figsize=(10, 6))
    plt.barh(d["feature"], d["shap_signed"])
    plt.axvline(0.0, linewidth=1.0)
    title = f"race={int(df_row['race_id'])} driver={int(df_row['driver_id'])} lap={int(df_row['lapno'])} | y={int(df_row['y_true'])} p={float(df_row['proba']):.3f} ({df_row['cohort']})"
    plt.title(title)
    plt.xlabel("Aggregated SHAP contribution (log-odds)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# -----------------------------
# main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="results/runs/runs_final/feb04", help="Frozen run directory (contains manifest.json + stage1_binary/artifacts).")
    ap.add_argument("--data_stage1", type=str, default="data/processed/output1.csv", help="Stage1 CSV containing ALL races (including holdouts).")
    ap.add_argument("--outdir", type=str, default="", help="Output directory. Default: <run_dir>/holdout_eval/shap_meta_stage1")
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for TP/FP/FN splits.")
    ap.add_argument("--per_race_cases", type=int, default=20, help="How many TP/FP/FN instances per race to generate local case-study plots for.")
    ap.add_argument("--top_k_features", type=int, default=12, help="Top-k features to export per instance.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    artifacts_dir = run_dir / "stage1_binary" / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Could not find artifacts dir: {artifacts_dir}")

    outdir = Path(args.outdir) if args.outdir else (run_dir / "holdout_eval" / "shap_meta_stage1")
    _ensure_dir(outdir)
    _ensure_dir(outdir / "plots")
    _ensure_dir(outdir / "case_studies")

    # load manifest for TCN config (seq_len/pad/mask)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    tcn_cfg = (manifest.get("stage1_binary", {}).get("tcn", {})) if isinstance(manifest, dict) else {}

    # load meta feature columns + X1 columns
    meta_features_path = artifacts_dir / "meta_features.json"
    feature_columns_path = artifacts_dir / "feature_columns.json"
    if not meta_features_path.exists() or not feature_columns_path.exists():
        raise FileNotFoundError(f"Missing meta_features.json or feature_columns.json in {artifacts_dir}")

    meta_cols = read_json(meta_features_path)["columns"]
    X1_cols = read_json(feature_columns_path)["X1_columns"]

    vprint(args.verbose, f"[run] run_dir={run_dir}")
    vprint(args.verbose, f"[run] outdir={outdir}")
    vprint(args.verbose, f"[run] holdouts={list(map(int, HOLDOUT_RACE_IDS))}")
    vprint(args.verbose, f"[run] meta_cols={len(meta_cols)} X1_cols={len(X1_cols)}")

    # load holdout data
    df_h = load_holdout_df(Path(args.data_stage1), verbose=args.verbose)

    # build base feature frame to match training
    X_base = build_base_feature_frame(df_h, X1_cols=X1_cols)

    # base probs
    base_probs = predict_base_probs(artifacts_dir=artifacts_dir, X_base=X_base, verbose=args.verbose)

    # TCN features
    tcn_proba, tcn_eff_scaled = predict_tcn_features(
        artifacts_dir=artifacts_dir,
        df_h=df_h,
        tcn_cfg=tcn_cfg,
        verbose=args.verbose,
    )

    # meta df
    X_meta = build_meta_df(
        df_h=df_h,
        meta_cols=meta_cols,
        base_probs=base_probs,
        tcn_proba=tcn_proba,
        tcn_eff_scaled=tcn_eff_scaled,
    )

    # load meta pipeline
    meta_pipe_path = artifacts_dir / "meta_pipeline.joblib"
    if not meta_pipe_path.exists():
        raise FileNotFoundError(f"Missing meta pipeline: {meta_pipe_path}")
    meta_pipe = load(meta_pipe_path)

    # predict
    proba = meta_pipe.predict_proba(X_meta)[:, 1].astype(float)
    y_true = df_h["y_pit"].to_numpy().astype(int)

    thr = float(args.threshold)
    cohort = assign_cohort(y_true, proba, thr)

    df_info = df_h[["race_id", "driver_id", "lapno"]].copy()
    if "row_id" in df_h.columns:
        df_info["row_id"] = df_h["row_id"].astype(int)
    else:
        df_info["row_id"] = np.arange(len(df_h), dtype=int)

    df_info["y_true"] = y_true
    df_info["proba"] = proba
    df_info["cohort"] = cohort

    # overall metrics on holdouts
    overall = metrics_binary(y_true, proba, thr=thr)
    (outdir / "holdout_metrics.json").write_text(json.dumps(overall, indent=2))
    print("[SHAP_eval_b] holdout metrics:", json.dumps(overall, indent=2), flush=True)

    # pick case studies (subset for local plots + per-instance export)
    df_cases = select_case_studies(df_info, per_race=int(args.per_race_cases), thr=thr).reset_index(drop=True)
    vprint(args.verbose, f"[cases] selected {len(df_cases)} instances for local explanations")

    # compute SHAP on the selected instances
    X_meta_cases = X_meta.loc[df_cases.index if X_meta.index.equals(df_info.index) else df_cases.index]
    # NOTE: df_cases is built from df_info rows; align by row_id index in a safer way
    # We'll align using original df_info position:
    pos_map = {int(rid): i for i, rid in enumerate(df_info["row_id"].to_list())}
    case_positions = [pos_map[int(rid)] for rid in df_cases["row_id"].to_list()]
    X_meta_cases = X_meta.iloc[case_positions].reset_index(drop=True)

    shap_vals, expected_value, pre, model, Xt_cases = compute_shap_tree(
        meta_pipe=meta_pipe, X_meta=X_meta_cases, verbose=args.verbose
    )

    # transformed feature names + mapping to original
    transformed_names = get_feature_names_out(pre, X_meta_cases)
    shap_signed_orig, shap_abs_orig = aggregate_shap_to_original(
        shap_vals=shap_vals,
        transformed_names=transformed_names,
        X_meta=X_meta_cases,
    )

    # build contributor tables
    df_cases_info = df_info.iloc[case_positions].reset_index(drop=True)
    top_k = int(args.top_k_features)

    contrib_long = top_contributors_long(
        df_info=df_cases_info,
        shap_signed_orig=shap_signed_orig,
        shap_abs_orig=shap_abs_orig,
        X_meta=X_meta_cases,
        top_k=top_k,
    )
    contrib_long.to_csv(outdir / "per_instance_top_contributors.csv", index=False)

    # cohort importance (from selected cases)
    imp = cohort_importance(df_cases_info, shap_abs_orig)
    imp.to_csv(outdir / "cohort_mean_abs_shap.csv", index=False)

    # grouped attribution (race x cohort shares)
    grp = grouped_attribution(df_cases_info, shap_abs_orig)
    grp.to_csv(outdir / "grouped_attribution_by_race_and_cohort.csv", index=False)

    # plots: cohort bars
    for c in ["TP", "FP", "FN", "TN"]:
        plot_cohort_bar(outdir / "plots" / f"bar_mean_abs_shap_{c}.png", imp, cohort=c, top_n=20)

    # local case-study plots per race
    case_dir = _ensure_dir(outdir / "case_studies")
    for race_id, g in df_cases_info.groupby("race_id", sort=True):
        race_dir = _ensure_dir(case_dir / f"race_{int(race_id)}")
        for _, row in g.iterrows():
            rid = int(row["row_id"])
            # get contributions for this instance
            d = contrib_long[contrib_long["row_id"] == rid]
            if len(d) == 0:
                continue
            fname = f"row_{rid}_lap_{int(row['lapno'])}_{row['cohort']}.png"
            plot_instance_contrib(race_dir / fname, row, d, top_n=min(10, top_k))

    # print a quick “what drives mistakes” summary (selected set)
    def top_features_for_cohort(coh: str, n: int = 8) -> List[Tuple[str, float]]:
        d = imp[imp["cohort"] == coh].sort_values("mean_abs_shap", ascending=False).head(n)
        return list(zip(d["feature"].tolist(), d["mean_abs_shap"].tolist()))

    print("\n[SHAP_eval_b] Top features by cohort (mean |SHAP| on selected cases):", flush=True)
    for coh in ["TP", "FP", "FN"]:
        tops = top_features_for_cohort(coh, n=10)
        print(f"  {coh}: " + ", ".join(f"{f}({v:.3f})" for f, v in tops), flush=True)

    print(f"\n[SHAP_eval_b] wrote outputs to: {outdir}", flush=True)

if __name__ == "__main__":
    main()