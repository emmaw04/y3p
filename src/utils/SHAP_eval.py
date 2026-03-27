import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from joblib import load

from src.data.data import HOLDOUT_RACE_IDS

# hardcoded paths for the final holdout evaluation
STAGE1_CSV = Path("data/processed/dataset1.csv")
STAGE2_CSV = Path("data/processed/dataset2.csv")
ARTIFACTS_ROOT = Path("runs/final_run")
OUTDIR = Path("runs/holdout_run/shap_eval")
PIT_THRESHOLD = 0.3120

def _make_left_padded_sequences_from_rows(
    keys_df: pd.DataFrame, X_row: np.ndarray, window: int, add_timestep_mask: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    constructs 3d sequences required for the recurrent neural networks
    groups by race and driver to prevent overlapping sequences
    """
    X_row = np.asarray(X_row)
    N, F = X_row.shape
    
    # if masking is enabled we need an extra feature column at the end
    if add_timestep_mask:
        X_seq = np.zeros((N, window, F + 1), dtype=np.float32)
    else:
        X_seq = np.zeros((N, window, F), dtype=np.float32)
        
    eff_len = np.zeros((N,), dtype=np.float32)
    group_indices = keys_df.groupby(["race_id", "driver_id"], sort=False).indices

    for _, idx in group_indices.items():
        idx = np.asarray(idx, dtype=int)
        for t, i in enumerate(idx):
            start = max(0, t - window + 1)
            take = idx[start : t + 1]
            L = len(take)
            
            if add_timestep_mask:
                X_seq[i, -L:, :F] = X_row[take, :]
                X_seq[i, -L:, F] = 1.0  # flag as real data not padding
            else:
                X_seq[i, -L:, :] = X_row[take, :]
                
            eff_len[i] = L
    return X_seq, eff_len

def _safe_predict_proba(pipe, X: pd.DataFrame) -> np.ndarray:
    """
    wrapper around sklearn predict proba to ensure we always get a float array back
    """
    proba = pipe.predict_proba(X)
    return np.asarray(proba, dtype=float)

def _keras_predict_proba(model: tf.keras.Model, X_np: np.ndarray) -> np.ndarray:
    """
    grabs probability of the positive class from a binary keras model
    handles different output shapes just in case
    """
    y = model.predict(X_np, batch_size=4096, verbose=0)
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0].astype(float)
    if y.ndim == 2 and y.shape[1] >= 2:
        return y[:, 1].astype(float)
    if y.ndim == 1:
        return y.astype(float)
    raise ValueError(f"unexpected keras shape: {y.shape}")

def _keras_predict_proba_multi(model: tf.keras.Model, X_np: np.ndarray) -> np.ndarray:
    """
    grabs probabilities for all classes from a multiclass keras model
    """
    y = model.predict(X_np, batch_size=4096, verbose=0)
    return np.asarray(y).astype(float)

def get_pipeline_model(pipe):
    """
    extracts the final estimator step out of a pipeline
    useful for passing the bare model to the shap explainer
    """
    if hasattr(pipe, "named_steps"):
        return pipe.named_steps["model"]
    return pipe

def evaluate_stage1():
    """
    runs tree shap on the stage 1 binary meta model
    calculates feature importance for pit stop prediction
    """
    print("running stage 1 shap evaluation")
    df1 = pd.read_csv(STAGE1_CSV)
    
    # clean up identifiers to avoid weird float issues
    for c in ["race_id", "driver_id", "lapno"]:
        df1[c] = pd.to_numeric(df1[c], errors="coerce").astype("Int64")
    
    # slice down to just the holdout set and sort
    holdout_set = set(int(x) for x in HOLDOUT_RACE_IDS)
    dfh = df1[df1["race_id"].isin(list(holdout_set))].copy()
    dfh = dfh.sort_values(["race_id", "driver_id", "lapno"], kind="mergesort").reset_index(drop=True)
    
    stage1_art = ARTIFACTS_ROOT / "stage1_binary" / "artifacts"
    x1_cols = json.loads((stage1_art / "feature_columns.json").read_text())["columns"]

    # fill in any missing features to prevent crashes during transform
    for c in x1_cols:
        if c not in dfh.columns:
            dfh[c] = 0
            
    base_svm = load(stage1_art / "base_svm_pipeline.joblib")
    base_xgb = load(stage1_art / "base_xgb_pipeline.joblib")
    lstm_pre = load(stage1_art / "lstm_preprocessor.joblib")
    lstm_model = tf.keras.models.load_model(stage1_art / "lstm_model.keras", compile=False)
    tcn_gru_pre = load(stage1_art / "tcn_gru_preprocessor.joblib")
    tcn_gru_model = tf.keras.models.load_model(stage1_art / "tcn_gru_model.keras", compile=False)
    meta_pipe = load(stage1_art / "meta_pipeline.joblib")

    X_raw = dfh[x1_cols].copy()
    
    # some models expect weather features to be floats
    for c in ["is_wet_race", "is_raining", "minutes_rain"]:
        if c in X_raw.columns:
            X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce").fillna(0.0)
        else:
            X_raw[c] = 0.0

    # collect base probabilities for tabular models
    p_svm = _safe_predict_proba(base_svm, X_raw)[:, 1]
    p_xgb = _safe_predict_proba(base_xgb, X_raw)[:, 1]
    keys_df = dfh[["race_id", "driver_id", "lapno"]]

    # preprocess and build sequences for lstm
    X_lstm_row = lstm_pre.transform(X_raw)
    if hasattr(X_lstm_row, "toarray"): X_lstm_row = X_lstm_row.toarray()
    X_seq_lstm, _ = _make_left_padded_sequences_from_rows(keys_df, X_lstm_row, window=lstm_model.input_shape[1], add_timestep_mask=True)
    p_lstm = _keras_predict_proba(lstm_model, X_seq_lstm)

    # preprocess and build sequences for tcn gru
    X_tcn_gru_row = tcn_gru_pre.transform(X_raw)
    if hasattr(X_tcn_gru_row, "toarray"): X_tcn_gru_row = X_tcn_gru_row.toarray()
    X_seq_tcn_gru, _ = _make_left_padded_sequences_from_rows(keys_df, X_tcn_gru_row, window=tcn_gru_model.input_shape[1], add_timestep_mask=True)
    p_tcn_gru = _keras_predict_proba(tcn_gru_model, X_seq_tcn_gru)

    # stitch together base predictions into a new dataframe
    X_meta = X_raw.copy()
    X_meta["p_svm"] = p_svm
    X_meta["p_xgb"] = p_xgb
    X_meta["p_lstm"] = p_lstm
    X_meta["p_tcn_gru"] = p_tcn_gru
    
    # filter to only what the meta model was trained on
    meta_cols = json.loads((stage1_art / "meta_features.json").read_text())["columns"]
    X_meta = X_meta[meta_cols]

    # run the data through the preprocessor so tree shap gets numeric arrays
    meta_model = get_pipeline_model(meta_pipe)
    if hasattr(meta_pipe, "named_steps") and "preprocess" in meta_pipe.named_steps:
        preprocessor = meta_pipe.named_steps["preprocess"]
        X_meta_t = preprocessor.transform(X_meta)
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = [f"f{i}" for i in range(X_meta_t.shape[1])]
    else:
        X_meta_t = X_meta
        feature_names = X_meta.columns

    explainer = shap.TreeExplainer(meta_model)
    if hasattr(X_meta_t, "toarray"): X_meta_t = X_meta_t.toarray()
    shap_vals = explainer.shap_values(X_meta_t)
    
    # binary xgboost usually returns a list
    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    else:
        sv = shap_vals

    # calculate overall importance by taking the mean of absolute shap values
    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
    
    importance_df.to_csv(OUTDIR / "stage1_shap_importance.csv", index=False)
    print(f"saved stage 1 shap importances to {OUTDIR}/stage1_shap_importance.csv")

def evaluate_stage2():
    """
    runs tree shap on the stage 2 multiclass meta model
    calculates feature importance for each tire compound independently
    """
    print("running stage 2 shap evaluation")
    df2 = pd.read_csv(STAGE2_CSV)
    
    # clean up identifiers
    for c in ["race_id", "driver_id", "lapno"]:
        df2[c] = pd.to_numeric(df2[c], errors="coerce").astype("Int64")
        
    holdout_set = set(int(x) for x in HOLDOUT_RACE_IDS)
    dfh = df2[df2["race_id"].isin(list(holdout_set))].copy()
    dfh = dfh.sort_values(["race_id", "driver_id", "lapno"], kind="mergesort").reset_index(drop=True)
    
    stage2_art = ARTIFACTS_ROOT / "stage2_multiclass" / "artifacts"
    x2_cols = json.loads((stage2_art / "feature_columns.json").read_text())["columns"]
    classes = json.loads((stage2_art / "classes.json").read_text())["classes"]
    
    # plug missing columns with zeros so the transformer doesnt break
    for c in x2_cols:
        if c not in dfh.columns:
            dfh[c] = 0

    base_svm = load(stage2_art / "base_svm_pipeline.joblib")
    base_xgb = load(stage2_art / "base_xgb_pipeline.joblib")
    base_rf = load(stage2_art / "base_rf_pipeline.joblib")
    tcn_gru_pre = load(stage2_art / "tcn_gru_preprocessor.joblib")
    tcn_gru_model = tf.keras.models.load_model(stage2_art / "tcn_gru_model.keras", compile=False)
    meta_pipe = load(stage2_art / "meta_pipeline.joblib")

    X_raw = dfh[x2_cols].copy()
    for c in ["is_wet_race", "is_raining", "minutes_rain"]:
        if c in X_raw.columns:
            X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce").fillna(0.0)
        else:
            X_raw[c] = 0.0

    # get predictions for all base learners
    p_svm = _safe_predict_proba(base_svm, X_raw)
    p_xgb = _safe_predict_proba(base_xgb, X_raw)
    p_rf = _safe_predict_proba(base_rf, X_raw)
    
    # build sequences and predict for tcn gru
    keys_df = dfh[["race_id", "driver_id", "lapno"]]
    X_tcn_gru_row = tcn_gru_pre.transform(X_raw)
    if hasattr(X_tcn_gru_row, "toarray"): X_tcn_gru_row = X_tcn_gru_row.toarray()
    X_seq_tcn_gru, _ = _make_left_padded_sequences_from_rows(keys_df, X_tcn_gru_row, window=tcn_gru_model.input_shape[1], add_timestep_mask=True)
    p_tcn_gru = _keras_predict_proba_multi(tcn_gru_model, X_seq_tcn_gru)

    X_meta = X_raw.copy()
    n_classes = len(classes)
    
    # inject the probability of every class from every base learner
    for i in range(n_classes):
        X_meta[f"p_svm_c{i}"] = p_svm[:, i]
        X_meta[f"p_xgb_c{i}"] = p_xgb[:, i]
        X_meta[f"p_rf_c{i}"] = p_rf[:, i]
        X_meta[f"p_tcn_gru_c{i}"] = p_tcn_gru[:, i]
        
    meta_cols = json.loads((stage2_art / "meta_features.json").read_text())["columns"]
    X_meta = X_meta[meta_cols]

    # pull the preprocessor to transform features before giving them to shap
    meta_model = get_pipeline_model(meta_pipe)
    if hasattr(meta_pipe, "named_steps") and "preprocess" in meta_pipe.named_steps:
        preprocessor = meta_pipe.named_steps["preprocess"]
        X_meta_t = preprocessor.transform(X_meta)
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = [f"f{i}" for i in range(X_meta_t.shape[1])]
    else:
        X_meta_t = X_meta
        feature_names = X_meta.columns

    explainer = shap.TreeExplainer(meta_model)
    if hasattr(X_meta_t, "toarray"): X_meta_t = X_meta_t.toarray()
    shap_vals = explainer.shap_values(X_meta_t)
    
    # shap values for multiclass can return as a list or a 3d array
    # loop through each tire compound and save its importances
    if isinstance(shap_vals, list):
        for i, cname in enumerate(classes):
            sv = shap_vals[i]
            mean_abs_shap = np.abs(sv).mean(axis=0)
            importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
            importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
            importance_df.to_csv(OUTDIR / f"stage2_shap_importance_{cname}.csv", index=False)
    else:
        if shap_vals.ndim == 3 and shap_vals.shape[2] == n_classes:
            for i, cname in enumerate(classes):
                sv = shap_vals[:, :, i]
                mean_abs_shap = np.abs(sv).mean(axis=0)
                importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
                importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
                importance_df.to_csv(OUTDIR / f"stage2_shap_importance_{cname}.csv", index=False)
                
    print(f"saved stage 2 shap importances to {OUTDIR}")

def main():
    # make sure the directory actually exists before trying to write files
    OUTDIR.mkdir(parents=True, exist_ok=True)
    evaluate_stage1()
    evaluate_stage2()

if __name__ == "__main__":
    main()
