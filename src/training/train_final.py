#!/usr/bin/env python3
# src/train_final.py
"""
Train *frozen* final model artifacts (NO holdout evaluation here).

Now includes --verbose to show progress in terminal.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone

from src.data import (
    load_stage1_dataset,
    load_stage2_dataset,
    get_stage1_xy,
    get_stage2_xy,
    make_race_group_folds,
    infer_feature_types,
    COMPOUND_CLASSES,
    HOLDOUT_RACE_IDS,
    build_feature_sequences,
)
from src.preprocessing import make_preprocessor_for_model, build_preprocessor, PreprocessConfig
from src.models import (
    ModelConfig,
    build_model_pipeline,
    get_binary_base_learners,
    get_multiclass_base_learners,
    make_meta_binary_xgb,
    make_meta_multiclass_xgb,
    make_tcn_binary,
)

# -----------------------------
# Helpers: printing + I/O
# -----------------------------

def vprint(verbose: bool, *args, **kwargs) -> None:
    if verbose:
        print(*args, **kwargs, flush=True)

def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# -----------------------------
# Helpers: OOF collection (train-only)
# -----------------------------

def _collect_oof_base_preds_binary(
    X: pd.DataFrame,
    y: pd.Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    base_learners: Dict[str, Any],
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    n = len(y)
    model_names = list(base_learners.keys())
    meta_X = np.zeros((n, len(model_names)), dtype=float)

    num_cols, cat_cols = infer_feature_types(X)

    vprint(verbose, f"[stage1][oof][tabular] n={n} models={model_names} folds={len(folds)}")

    for fold, (tr_idx, va_idx) in enumerate(folds):
        vprint(verbose, f"[stage1][oof][tabular][fold {fold}] train={len(tr_idx)} valid={len(va_idx)}")
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va = X.iloc[va_idx]

        for j, name in enumerate(model_names):
            vprint(verbose, f"  -> fitting base '{name}' (fold {fold})")
            est = clone(base_learners[name])

            if name == "ann":
                pre = make_preprocessor_for_model("ann", num_cols=num_cols, cat_cols=cat_cols)
            elif name == "svm":
                pre = make_preprocessor_for_model("svm", num_cols=num_cols, cat_cols=cat_cols)
            elif name == "rf":
                pre = make_preprocessor_for_model("rf", num_cols=num_cols, cat_cols=cat_cols)
            elif name == "xgb":
                pre = make_preprocessor_for_model("xgb", num_cols=num_cols, cat_cols=cat_cols)
            else:
                pre = make_preprocessor_for_model(name, num_cols=num_cols, cat_cols=cat_cols)

            pipe = build_model_pipeline(pre, est)
            pipe.fit(X_tr, y_tr)

            proba_pos = pipe.predict_proba(X_va)[:, 1]
            meta_X[va_idx, j] = proba_pos

            vprint(verbose, f"     wrote {len(va_idx)} probs for '{name}' (mean p={float(np.mean(proba_pos)):.4f})")

    return meta_X, model_names


def _collect_oof_tcn_preds_binary(
    df1: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    *,
    cfg: ModelConfig,
    seq_len: int,
    pad_left: bool,
    add_timestep_mask: bool,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    y_full = df1["y_pit"].astype(int)
    keys = df1[["race_id", "driver_id", "lapno"]].copy()

    X_tab = df1.drop(columns=["y_pit", "y_compound", "race_id", "driver_id"], errors="ignore").copy()
    num_cols, cat_cols = infer_feature_types(X_tab)
    pre_proto = make_preprocessor_for_model("tcn", num_cols=num_cols, cat_cols=cat_cols)

    N = len(df1)
    oof_pred = np.full(N, np.nan, dtype=float)
    oof_eff_len = np.zeros(N, dtype=float)

    vprint(verbose, f"[stage1][oof][tcn] N={N} folds={len(folds)} seq_len={seq_len} pad_left={pad_left} tmask={add_timestep_mask}")

    for fold, (tr_idx, va_idx) in enumerate(folds):
        vprint(verbose, f"[stage1][oof][tcn][fold {fold}] train={len(tr_idx)} valid={len(va_idx)}")

        X_tr, y_tr = X_tab.iloc[tr_idx], y_full.iloc[tr_idx]

        pre = clone(pre_proto)
        pre.fit(X_tr, y_tr)

        Xt_all = pre.transform(X_tab)
        if hasattr(Xt_all, "toarray"):
            Xt_all = Xt_all.toarray()
        Xt_all = Xt_all.astype(np.float32)

        X_seq, y_seq, idx_last, seq_idx, eff_len = build_feature_sequences(
            keys,
            Xt_all,
            y_full,
            seq_len=seq_len,
            pad_left=pad_left,
            add_timestep_mask=add_timestep_mask,
        )

        if pad_left:
            tr_mask = np.all((seq_idx == -1) | np.isin(seq_idx, tr_idx), axis=1)
            va_mask = np.all((seq_idx == -1) | np.isin(seq_idx, va_idx), axis=1)
        else:
            tr_mask = np.all(np.isin(seq_idx, tr_idx), axis=1)
            va_mask = np.all(np.isin(seq_idx, va_idx), axis=1)

        X_seq_tr, y_seq_tr = X_seq[tr_mask], y_seq[tr_mask]
        X_seq_va = X_seq[va_mask]
        idx_last_va = idx_last[va_mask]
        eff_len_va = eff_len[va_mask]

        vprint(verbose, f"  sequences: train={len(X_seq_tr)} valid={len(X_seq_va)} (written to {len(idx_last_va)} rows)")

        if len(y_seq_tr) == 0 or len(X_seq_va) == 0:
            vprint(verbose, f"  -> skipping fold {fold} (no sequences after filtering)")
            continue

        n_pos = int(np.sum(y_seq_tr == 1))
        n_neg = int(np.sum(y_seq_tr == 0))
        if n_pos == 0 or n_neg == 0:
            const_p = 1.0 if n_neg == 0 else 0.0
            oof_pred[idx_last_va] = const_p
            oof_eff_len[idx_last_va] = eff_len_va
            vprint(verbose, f"  -> degenerate labels (n_pos={n_pos}, n_neg={n_neg}), wrote const_p={const_p}")
            continue

        pos_w = min(20.0, float(n_neg / n_pos))
        class_w = {0: 1.0, 1: pos_w}
        vprint(verbose, f"  -> training TCN (pos_w={pos_w:.2f})")

        tcn = make_tcn_binary(cfg, seq_len=seq_len)
        tcn.fit(X_seq_tr, y_seq_tr, class_weight=class_w)

        proba_pos = tcn.predict_proba(X_seq_va)[:, 1]
        oof_pred[idx_last_va] = proba_pos
        oof_eff_len[idx_last_va] = eff_len_va
        vprint(verbose, f"  -> wrote {len(idx_last_va)} TCN probs (mean p={float(np.mean(proba_pos)):.4f})")

    oof_pred = np.nan_to_num(oof_pred, nan=0.0)
    eff_scaled = oof_eff_len / float(seq_len)
    return oof_pred, eff_scaled


def _meta_feature_df_stage1(
    meta_tab: np.ndarray,
    base_names: List[str],
    X_context: pd.DataFrame,
    tcn_proba: np.ndarray,
    tcn_eff_scaled: np.ndarray,
) -> pd.DataFrame:
    prob_cols = [f"p_{n}" for n in base_names]
    df_probs = pd.DataFrame(meta_tab, columns=prob_cols)
    df = pd.concat([df_probs.reset_index(drop=True), X_context.reset_index(drop=True)], axis=1)
    df["tcn_proba"] = tcn_proba
    df["tcn_effective_len"] = tcn_eff_scaled
    return df


def _collect_oof_base_preds_multiclass(
    X: pd.DataFrame,
    y: pd.Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    base_learners: Dict[str, Any],
    n_classes: int,
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    n = len(y)
    model_names = list(base_learners.keys())
    meta_X = np.zeros((n, len(model_names) * n_classes), dtype=float)

    num_cols, cat_cols = infer_feature_types(X)

    vprint(verbose, f"[stage2][oof][tabular] n={n} models={model_names} K={n_classes} folds={len(folds)}")

    for fold, (tr_idx, va_idx) in enumerate(folds):
        vprint(verbose, f"[stage2][oof][fold {fold}] train={len(tr_idx)} valid={len(va_idx)}")

        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va = X.iloc[va_idx]

        for j, name in enumerate(model_names):
            vprint(verbose, f"  -> fitting base '{name}' (fold {fold})")
            est = clone(base_learners[name])

            if name in {"ann", "vse_compound_ann"}:
                pre = make_preprocessor_for_model("ann", num_cols=num_cols, cat_cols=cat_cols)
            elif name == "svm":
                pre = make_preprocessor_for_model("svm", num_cols=num_cols, cat_cols=cat_cols)
            elif name == "rf":
                pre = make_preprocessor_for_model("rf", num_cols=num_cols, cat_cols=cat_cols)
            elif name == "xgb":
                pre = make_preprocessor_for_model("xgb", num_cols=num_cols, cat_cols=cat_cols)
            else:
                pre = make_preprocessor_for_model(name, num_cols=num_cols, cat_cols=cat_cols)

            pipe = build_model_pipeline(pre, est)
            pipe.fit(X_tr, y_tr)

            proba_fold = pipe.predict_proba(X_va)

            if hasattr(pipe, "named_steps") and "model" in pipe.named_steps and hasattr(pipe.named_steps["model"], "classes_"):
                classes_seen = pipe.named_steps["model"].classes_
            elif hasattr(pipe, "classes_"):
                classes_seen = pipe.classes_
            else:
                raise RuntimeError("Could not determine classes_ for multiclass alignment.")

            proba_full = np.zeros((len(va_idx), n_classes), dtype=float)
            for jj, cls in enumerate(classes_seen):
                proba_full[:, int(cls)] = proba_fold[:, jj]

            start = j * n_classes
            meta_X[va_idx, start:start + n_classes] = proba_full

            vprint(verbose, f"     wrote {len(va_idx)} prob vecs for '{name}' (mean maxp={float(np.mean(np.max(proba_full, axis=1))):.4f})")

    return meta_X, model_names


def _meta_feature_df_stage2(
    meta_X: np.ndarray,
    base_names: List[str],
    n_classes: int,
    X_context: pd.DataFrame,
) -> pd.DataFrame:
    cols: List[str] = []
    for name in base_names:
        for k in range(n_classes):
            cols.append(f"p_{name}_c{k}")
    df_probs = pd.DataFrame(meta_X, columns=cols)
    df = pd.concat([df_probs.reset_index(drop=True), X_context.reset_index(drop=True)], axis=1)
    return df

# -----------------------------
# Train final artifacts
# -----------------------------

def train_stage1_final(
    df1: pd.DataFrame,
    X1: pd.DataFrame,
    y1: pd.Series,
    folds1: List[Tuple[np.ndarray, np.ndarray]],
    outdir: Path,
    *,
    cfg: ModelConfig,
    meta_name: str = "xgb",
    tcn_seq_len: int = 8,
    tcn_pad_left: bool = True,
    tcn_add_timestep_mask: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    stage_dir = _ensure_dir(outdir / "stage1_binary")
    art_dir = _ensure_dir(stage_dir / "artifacts")

    base = get_binary_base_learners(cfg)
    base_names = list(base.keys())
    vprint(verbose, f"[stage1] base learners: {base_names}")

    vprint(verbose, "[stage1] collecting OOF tabular base predictions...")
    meta_tab, base_names = _collect_oof_base_preds_binary(X1, y1, folds1, base, verbose=verbose)

    vprint(verbose, "[stage1] collecting OOF TCN predictions...")
    tcn_oof, tcn_eff_scaled = _collect_oof_tcn_preds_binary(
        df1,
        folds1,
        cfg=cfg,
        seq_len=tcn_seq_len,
        pad_left=tcn_pad_left,
        add_timestep_mask=tcn_add_timestep_mask,
        verbose=verbose,
    )

    vprint(verbose, "[stage1] building meta feature dataframe...")
    X_meta = _meta_feature_df_stage1(meta_tab, base_names, X1, tcn_oof, tcn_eff_scaled)
    meta_feature_columns = list(X_meta.columns)
    _write_json(art_dir / "meta_features.json", {"columns": meta_feature_columns})

    vprint(verbose, f"[stage1] fitting final meta pipeline on ALL train rows (meta='{meta_name}')...")
    num_m, cat_m = infer_feature_types(X_meta)
    meta_pre = build_preprocessor(
        num_cols=num_m,
        cat_cols=cat_m,
        cfg=PreprocessConfig(scale_numeric=False, sparse_onehot=True),
    )
    meta_est = make_meta_binary_xgb(cfg)
    meta_pipe = build_model_pipeline(meta_pre, meta_est)
    meta_pipe.fit(X_meta, y1.to_numpy())
    dump(meta_pipe, art_dir / "meta_pipeline.joblib")

    vprint(verbose, "[stage1] retraining and freezing tabular base pipelines on ALL train rows...")
    num1, cat1 = infer_feature_types(X1)
    base_frozen: Dict[str, Any] = {}

    for name in base_names:
        est = clone(base[name])

        if name == "ann":
            vprint(verbose, f"  -> training base '{name}' on full train (keras-safe save)...")

            pre = make_preprocessor_for_model("ann", num_cols=num1, cat_cols=cat1)
            pre.fit(X1, y1)
            Xt = pre.transform(X1)
            if hasattr(Xt, "toarray"):
                Xt = Xt.toarray()
            Xt = Xt.astype(np.float32)

            ann = clone(base[name])
            ann.fit(Xt, y1.to_numpy())

            dump(pre, art_dir / "ann_preprocessor.joblib")

            ann_model_path = art_dir / "ann_model.keras"
            keras_model = getattr(ann, "model_", None) or getattr(ann, "model", None)
            if keras_model is None:
                raise RuntimeError("ANN wrapper has no .model_/.model attribute to save.")
            keras_model.save(ann_model_path)

            _write_json(art_dir / "ann_spec.json", {
                "type": "keras_binary_ann",
                "input_dim": int(Xt.shape[1]),
                "preprocessor": "ann_preprocessor.joblib",
                "model": "ann_model.keras",
                "uses_sparse_toarray": True,
            })

            base_frozen[name] = {
                "kind": "keras",
                "preprocessor": "ann_preprocessor.joblib",
                "model": "ann_model.keras",
                "spec": "ann_spec.json",
            }

            vprint(verbose, "     saved ann_preprocessor.joblib + ann_model.keras + ann_spec.json")
            continue

        # non-ANN models: save sklearn pipeline normally
        vprint(verbose, f"  -> training base '{name}' on full train...")
        pre = make_preprocessor_for_model(name, num_cols=num1, cat_cols=cat1)
        pipe = build_model_pipeline(pre, est)
        pipe.fit(X1, y1)

        path = art_dir / f"base_{name}_pipeline.joblib"
        dump(pipe, path)
        base_frozen[name] = path.name
        vprint(verbose, f"     saved {path.name}")


    vprint(verbose, "[stage1] retraining and freezing TCN on ALL train sequences...")
    X_tab = df1.drop(columns=["y_pit", "y_compound", "race_id", "driver_id"], errors="ignore").copy()
    y_full = df1["y_pit"].astype(int)
    keys = df1[["race_id", "driver_id", "lapno"]].copy()

    num_t, cat_t = infer_feature_types(X_tab)
    tcn_pre = make_preprocessor_for_model("tcn", num_cols=num_t, cat_cols=cat_t)
    tcn_pre.fit(X_tab, y_full)

    Xt_all = tcn_pre.transform(X_tab)
    if hasattr(Xt_all, "toarray"):
        Xt_all = Xt_all.toarray()
    Xt_all = Xt_all.astype(np.float32)

    X_seq, y_seq, idx_last, seq_idx, eff_len = build_feature_sequences(
        keys,
        Xt_all,
        y_full,
        seq_len=tcn_seq_len,
        pad_left=tcn_pad_left,
        add_timestep_mask=tcn_add_timestep_mask,
    )

    n_pos = int(np.sum(y_seq == 1))
    n_neg = int(np.sum(y_seq == 0))
    pos_w = 1.0
    if n_pos > 0 and n_neg > 0:
        pos_w = min(20.0, float(n_neg / n_pos))
    class_w = {0: 1.0, 1: float(pos_w)}

    vprint(verbose, f"  -> TCN train sequences={len(y_seq)} (pos={n_pos} neg={n_neg} pos_w={pos_w:.2f})")
    tcn = make_tcn_binary(cfg, seq_len=tcn_seq_len)
    tcn.fit(X_seq, y_seq, class_weight=class_w)

    dump(tcn_pre, art_dir / "tcn_preprocessor.joblib")
    tcn_model_path = art_dir / "tcn_model.keras"
    keras_model = getattr(tcn, "model_", None) or getattr(tcn, "model", None)
    if keras_model is None:
        raise RuntimeError("TCN wrapper has no .model_/.model attribute to save.")
    keras_model.save(tcn_model_path)
    vprint(verbose, f"     saved tcn_preprocessor.joblib and tcn_model.keras")

    _write_json(art_dir / "feature_columns.json", {"X1_columns": list(X1.columns)})
    vprint(verbose, "[stage1] saved feature_columns.json + meta_features.json")

    return {
        "stage": "stage1_binary",
        "base_models": base_names,
        "meta_model": meta_name,
        "X_columns": list(X1.columns),
        "meta_feature_columns": meta_feature_columns,
        "tcn": {
            "enabled": True,
            "seq_len": int(tcn_seq_len),
            "pad_left": bool(tcn_pad_left),
            "add_timestep_mask": bool(tcn_add_timestep_mask),
            "pad_value": -1.0,
            "effective_len_scaled_by": int(tcn_seq_len),
            "artifacts": {
                "preprocessor": "tcn_preprocessor.joblib",
                "model": "tcn_model.keras",
            },
        },
        "artifacts": {
            "meta_pipeline": "meta_pipeline.joblib",
            "meta_features": "meta_features.json",
            "feature_columns": "feature_columns.json",
            "base_pipelines": base_frozen,
        },
    }


def train_stage2_final(
    df2: pd.DataFrame,
    X2: pd.DataFrame,
    y2: pd.Series,
    folds2: List[Tuple[np.ndarray, np.ndarray]],
    outdir: Path,
    *,
    cfg: ModelConfig,
    meta_name: str = "xgb",
    verbose: bool = False,
) -> Dict[str, Any]:
    stage_dir = _ensure_dir(outdir / "stage2_multiclass")
    art_dir = _ensure_dir(stage_dir / "artifacts")

    n_classes = len(COMPOUND_CLASSES)
    base = get_multiclass_base_learners(cfg, n_classes=n_classes)
    base_names = list(base.keys())
    vprint(verbose, f"[stage2] base learners: {base_names} (K={n_classes})")

    y2_int = y2.astype(int)

    vprint(verbose, "[stage2] collecting OOF base predictions...")
    meta_X, base_names = _collect_oof_base_preds_multiclass(
        X2, y2_int, folds2, base, n_classes, verbose=verbose
    )

    vprint(verbose, "[stage2] building meta feature dataframe...")
    X_meta = _meta_feature_df_stage2(meta_X, base_names, n_classes, X2)
    meta_feature_columns = list(X_meta.columns)
    _write_json(art_dir / "meta_features.json", {"columns": meta_feature_columns})

    vprint(verbose, f"[stage2] fitting final meta pipeline on ALL train rows (meta='{meta_name}')...")
    num_m, cat_m = infer_feature_types(X_meta)
    meta_pre = build_preprocessor(
        num_cols=num_m,
        cat_cols=cat_m,
        cfg=PreprocessConfig(scale_numeric=False, sparse_onehot=True),
    )
    meta_est = make_meta_multiclass_xgb(cfg, n_classes=n_classes)
    meta_pipe = build_model_pipeline(meta_pre, meta_est)
    meta_pipe.fit(X_meta, y2_int.to_numpy())
    dump(meta_pipe, art_dir / "meta_pipeline.joblib")

    vprint(verbose, "[stage2] retraining and freezing base pipelines on ALL train rows...")
    num2, cat2 = infer_feature_types(X2)
    base_frozen: Dict[str, Any] = {}

    for name in base_names:
        est = clone(base[name])

        if name in {"ann", "vse_compound_ann"}:
            vprint(verbose, f"  -> training base '{name}' on full train (keras-safe save)...")

            pre = make_preprocessor_for_model("ann", num_cols=num2, cat_cols=cat2)
            pre.fit(X2, y2_int)
            Xt = pre.transform(X2)
            if hasattr(Xt, "toarray"):
                Xt = Xt.toarray()
            Xt = Xt.astype(np.float32)

            ann = clone(base[name])
            ann.fit(Xt, y2_int.to_numpy())

            dump(pre, art_dir / f"{name}_preprocessor.joblib")

            ann_model_path = art_dir / f"{name}_model.keras"
            keras_model = getattr(ann, "model_", None) or getattr(ann, "model", None)
            if keras_model is None:
                raise RuntimeError(f"{name} wrapper has no .model_/.model attribute to save.")
            keras_model.save(ann_model_path)

            spec_name = f"{name}_spec.json"
            _write_json(art_dir / spec_name, {
                "type": "keras_multiclass_ann",
                "n_classes": int(n_classes),
                "input_dim": int(Xt.shape[1]),
                "preprocessor": f"{name}_preprocessor.joblib",
                "model": f"{name}_model.keras",
                "uses_sparse_toarray": True,
            })

            base_frozen[name] = {
                "kind": "keras",
                "preprocessor": f"{name}_preprocessor.joblib",
                "model": f"{name}_model.keras",
                "spec": spec_name,
            }

            vprint(verbose, f"     saved {name}_preprocessor.joblib + {name}_model.keras + {spec_name}")
            continue

        # non-ANN models
        vprint(verbose, f"  -> training base '{name}' on full train...")
        pre_key = "ann" if name == "vse_compound_ann" else name
        pre = make_preprocessor_for_model(pre_key, num_cols=num2, cat_cols=cat2)
        pipe = build_model_pipeline(pre, est)
        pipe.fit(X2, y2_int)

        path = art_dir / f"base_{name}_pipeline.joblib"
        dump(pipe, path)
        base_frozen[name] = path.name
        vprint(verbose, f"     saved {path.name}")

    _write_json(art_dir / "feature_columns.json", {"X2_columns": list(X2.columns)})
    _write_json(art_dir / "classes.json", {
        "classes": list(COMPOUND_CLASSES),
        "class_to_int": {c: i for i, c in enumerate(COMPOUND_CLASSES)},
        "int_to_class": {i: c for i, c in enumerate(COMPOUND_CLASSES)},
    })
    vprint(verbose, "[stage2] saved feature_columns.json + meta_features.json + classes.json")

    return {
        "stage": "stage2_multiclass",
        "n_classes": int(n_classes),
        "base_models": base_names,
        "meta_model": meta_name,
        "X_columns": list(X2.columns),
        "meta_feature_columns": meta_feature_columns,
        "classes": list(COMPOUND_CLASSES),
        "artifacts": {
            "meta_pipeline": "meta_pipeline.joblib",
            "meta_features": "meta_features.json",
            "feature_columns": "feature_columns.json",
            "classes": "classes.json",
            "base_pipelines": base_frozen,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_stage1", required=True, help="Path to stage1 dataset (pit/no-pit)")
    ap.add_argument("--data_stage2", required=True, help="Path to stage2 dataset (pit-stop-only compound)")
    ap.add_argument("--outdir", default="runs_final", help="Output root for frozen artifacts")
    ap.add_argument("--run_name", default="final_run", help="Subfolder name under outdir")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--meta_stage1", choices=["xgb"], default="xgb")
    ap.add_argument("--meta_stage2", choices=["xgb"], default="xgb")

    ap.add_argument("--tcn_seq_len", type=int, default=8)
    ap.add_argument("--tcn_pad_left", action="store_true", default=True)
    ap.add_argument("--tcn_no_pad_left", action="store_true", default=False)
    ap.add_argument("--tcn_add_timestep_mask", action="store_true", default=True)
    ap.add_argument("--tcn_no_timestep_mask", action="store_true", default=False)

    ap.add_argument("--only_stage", choices=["all", "stage1", "stage2"], default="all")
    ap.add_argument("--verbose", action="store_true", help="Print progress logs")
    args = ap.parse_args()

    # resolve booleans cleanly
    tcn_pad_left = True if args.tcn_pad_left else False
    if args.tcn_no_pad_left:
        tcn_pad_left = False

    tcn_tmask = True if args.tcn_add_timestep_mask else False
    if args.tcn_no_timestep_mask:
        tcn_tmask = False

    root = _ensure_dir(Path(args.outdir) / args.run_name)
    cfg = ModelConfig(random_state=args.seed)

    vprint(args.verbose, f"[train_final] output root: {root}")
    vprint(args.verbose, f"[train_final] seed={args.seed} n_splits={args.n_splits}")
    vprint(args.verbose, f"[train_final] holdout_race_ids={list(map(int, HOLDOUT_RACE_IDS))}")

    # Load datasets (train-only)
    vprint(args.verbose, "[train_final] loading datasets (train-only; holdouts excluded in loader)...")
    df1 = load_stage1_dataset(args.data_stage1)
    df2 = load_stage2_dataset(args.data_stage2, strict=True)
    vprint(args.verbose, f"[train_final] df1 rows={len(df1)} df2 rows={len(df2)}")

    # X/y
    vprint(args.verbose, "[train_final] building X/y...")
    X1, y1 = get_stage1_xy(df1)
    X2, y2 = get_stage2_xy(df2)
    vprint(args.verbose, f"[train_final] stage1 X1={X1.shape} y1={y1.shape}")
    vprint(args.verbose, f"[train_final] stage2 X2={X2.shape} y2={y2.shape}")

    # Folds
    vprint(args.verbose, "[train_final] building race-grouped folds...")
    fb1 = make_race_group_folds(df1, target_col="y_pit", n_splits=args.n_splits, seed=args.seed)
    folds1 = fb1.folds
    vprint(args.verbose, f"[train_final] stage1 folds={len(folds1)}")

    df2_tmp = df2.copy()
    df2_tmp = df2_tmp.assign(y_compound_encoded=y2.to_numpy())
    fb2 = make_race_group_folds(df2_tmp, target_col="y_compound_encoded", n_splits=args.n_splits, seed=args.seed)
    folds2 = fb2.folds
    vprint(args.verbose, f"[train_final] stage2 folds={len(folds2)}")

    manifest: Dict[str, Any] = {
        "run_name": args.run_name,
        "seed": int(args.seed),
        "n_splits": int(args.n_splits),
        "holdout_race_ids": list(map(int, HOLDOUT_RACE_IDS)),
        "datasets": {
            "stage1_path": str(args.data_stage1),
            "stage2_path": str(args.data_stage2),
            "note": "Loaders excluded holdout races at load time; final models are trained on non-holdout races only.",
        },
        "model_config": asdict(cfg),
    }

    # Train stage1
    if args.only_stage in {"all", "stage1"}:
        vprint(args.verbose, "\n[train_final] === TRAIN STAGE 1 (binary) ===")
        stage1_info = train_stage1_final(
            df1,
            X1,
            y1,
            folds1,
            root,
            cfg=cfg,
            meta_name=args.meta_stage1,
            tcn_seq_len=args.tcn_seq_len,
            tcn_pad_left=tcn_pad_left,
            tcn_add_timestep_mask=tcn_tmask,
            verbose=args.verbose,
        )
        manifest["stage1_binary"] = stage1_info

    # Train stage2
    if args.only_stage in {"all", "stage2"}:
        vprint(args.verbose, "\n[train_final] === TRAIN STAGE 2 (multiclass) ===")
        stage2_info = train_stage2_final(
            df2,
            X2,
            y2,
            folds2,
            root,
            cfg=cfg,
            meta_name=args.meta_stage2,
            verbose=args.verbose,
        )
        manifest["stage2_multiclass"] = stage2_info

    _write_json(root / "manifest.json", manifest)

    print(f"[train_final] wrote frozen artifacts to: {root}", flush=True)
    print(f"[train_final] manifest: {root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()