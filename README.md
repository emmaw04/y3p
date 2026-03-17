# Hierarchical Stacking Ensemble for Formula 1 Race Strategy Prediction

This repository contains the implementation of a PhD-level research project focused on predicting Formula 1 race strategies using a two-stage hierarchical machine learning framework. The system integrates traditional tabular classifiers with deep temporal models (TCN, GRU) in a stacking ensemble architecture, optimized via Optuna and validated through race-grouped cross-validation to mitigate data leakage.

## Abstract

Race strategy in Formula 1 is a multi-objective optimization problem characterized by high stochasticity and temporal dependencies. This project decomposes the problem into two distinct classification tasks:
1.  **Stage 1 (Decision):** A binary classification task predicting the probability of a pit stop occurring on any given lap ($y_{pit} \in \{0, 1\}$).
2.  **Stage 2 (Selection):** A multiclass classification task predicting the optimal tire compound for the subsequent stint ($y_{compound} \in \{\text{Hard, Medium, Soft, Intermediate, Wet}\}$), conditioned on a pit stop event.

The architecture employs a **Meta-Stacking** approach where base learner Out-Of-Fold (OOF) probabilities are used as augmented features for a second-level Gradient Boosted Decision Tree (XGBoost) meta-learner.

---

## 📂 Recursive Directory Structure

### 📁 `src/` - Core Logic and Implementation
The `src` directory contains the modular implementation of the pipeline.
*   **`src/data/`**: Data orchestration and ingestion.
    *   `data.py`: Defines the rigorous schema for Stage 1/2 datasets, implements the `FoldBundle` logic for race-grouped cross-validation, and provides sequence builders for temporal models.
    *   `preprocessing.py`: Implements `scikit-learn` pipelines for heterogeneous data handling (One-Hot Encoding for categorical race tracks, Scaling for lap-time deltas).
*   **`src/models/`**: Architectural definitions.
    *   `models.py`: A factory module for diverse estimators. Includes implementations for Temporal Convolutional Networks (TCN) with causal dilations, GRU/LSTM recurrent units, and tuned configurations for Random Forests and SVMs.
*   **`src/training/`**: Execution and validation.
    *   `train_final.py`: Retrains the full hierarchical ensemble on the entire non-holdout dataset to produce frozen artifacts.
    *   `evaluate_hierarchical_ensemble.py`: End-to-end evaluation of the joint probability $P(y_{pit}, y_{compound})$.
    *   `evaluate_stack.py`: Specific evaluation of the stacking meta-learner performance.
*   **`src/utils/`**: Research and Optimization.
    *   `tune_optuna.py`: Hyperparameter optimization (HPO) logic using Tree-structured Parzen Estimator (TPE) samplers and Median Pruners.
    *   `SHAP_eval_b.py`: Interpretability module using SHAP (SHapley Additive exPlanations) to quantify feature attribution.
    *   `smote_generator.py`: Synthetic Minority Over-sampling Technique (SMOTE) for addressing class imbalance in Stage 2 wet-weather scenarios.

### 📁 `data/` - Data Persistence
*   **`data/raw/`**: Contains the source SQLite database (`fastf1_vse_plus_final.sqlite`) derived from FastF1 and timing data.
*   **`data/processed/`**: Cleaned, feature-engineered CSV/Parquet files.
    *   `output1.csv` / `output2.csv`: Primary Stage 1 and Stage 2 datasets.
    *   `SMOTE/`: Augmented datasets for multiclass imbalance handling.
    *   `pit_stops_left_dataset/`: Specialized features for stint-remaining regressions.
*   **`data/scripts/`**: ETL (Extract, Transform, Load) scripts for database cleaning and feature construction.

### 📁 `results/` & `runs/` - Experimental Artifacts
*   **`optunaruns/`**: JSON/CSV exports of optimization trials, capturing the evolution of hyperparameter search spaces.
*   **`tuning_8_final/`**: Final Optuna `.db` files and serialized best-parameter configurations.
*   **`ablation/`**: Results from feature ablation studies used to determine the impact of specific telemetry channels on predictive accuracy.

### 📁 `VSE/` - Virtual Strategy Engineer
A submodule containing a **Virtual Race Simulator**.
*   Used for "Closed-loop" validation: Testing the ML model's predicted strategies in a simulated environment to measure "Time-to-Finish" delta against real-world strategies.

---

## 🛠 Methodology

### Race-Grouped Cross-Validation
Standard k-fold CV is unsuitable for race data due to the high correlation between laps in the same event. We implement a **Grouped-Shuffle-Split** based on `race_id`. This ensures that if any lap of the "2023 Monaco GP" is in the validation set, *all* laps from that race are excluded from the training set, preventing temporal leakage.

### Stacking Ensemble Architecture
1.  **Base Layer:** SVM, Random Forest, XGBoost, ANN, and TCN are trained on raw features.
2.  **Meta-Feature Generation:** OOF probabilities are generated for each base learner.
3.  **Meta Layer:** An XGBoost meta-learner is trained on the concatenation of original features and base-learner probabilities.

---

## 🚀 Reproduction Workflow

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Hyperparameter Optimization
To optimize the Stage 1 TCN model (requires GPU/TensorFlow):
```bash
python -m src.utils.tune_optuna \
    --data_stage1 data/processed/output1.csv \
    --task binary \
    --model tcn \
    --n_trials 200 \
    --outdir runs/tuning_tcn
```

### 3. Training the Frozen Pipeline
```bash
python -m src.training.train_final \
    --data_stage1 data/processed/output1.csv \
    --data_stage2 data/processed/output2.csv \
    --run_name march_final_v1 \
    --verbose
```

### 4. SHAP Interpretability Analysis
```bash
python -m src.utils.SHAP_eval_b --model_path results/runs_final/stage1_meta.joblib
```

---

## 📊 Evaluation Metrics
*   **Stage 1:** F1-Score, PR-AUC (Priority due to class imbalance), and Brier Score (for calibration).
*   **Stage 2:** Macro-Averaged F1, Balanced Accuracy, and Top-2 Accuracy.

## 📝 Citation
If using this research in an academic context, please cite the associated PhD thesis:
*(Citation details to be finalized upon publication)*
