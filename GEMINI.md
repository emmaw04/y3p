# Project Overview

This project implements a sophisticated machine learning pipeline for Formula 1 race strategy prediction. It leverages a two-stage hierarchical approach to predict pit stop timing and tire compound choice.

1.  **Stage 1 (Binary Classification):** Predicts `y_pit` (whether a pit stop occurs on a given lap).
2.  **Stage 2 (Multiclass Classification):** Predicts `y_compound` (the tire compound of the next set fitted during a pit stop).

The system employs a **stacking ensemble** methodology, combining predictions from diverse base learners (Random Forest, XGBoost, SVM, ANN, TCN, GRU, LSTM) using a meta-learner (typically XGBoost). To prevent data leakage, it utilizes a robust **race-grouped cross-validation** strategy, ensuring entire races are kept together in training or validation folds.

## Project Structure

-   `data/`: Contains raw SQL data (`fastf1_vse_plus_final.sqlite`), processed CSVs (`output1.csv`, `output2.csv`), and SMOTE-oversampled datasets.
-   `src/`: Core source code.
    -   `data/`: Data loaders (`load_stage1_dataset`), preprocessing pipelines, and sequence builders.
    -   `models/`: Factory functions for base and meta-learners, including TCN and GRU architectures.
    -   `training/`: Scripts for evaluating models (`evaluate_stack.py`, `evaluate_base.py`) and training final frozen artifacts (`train_final.py`).
    -   `utils/`: Hyperparameter tuning via Optuna (`tune_optuna.py`), SHAP analysis, and ablation studies.
-   `VSE/`: A submodule for the "Virtual Strategy Engineer" environment, providing a race simulation core for strategy testing.
-   `results/` & `runs/`: Storage for experiment logs, Optuna study databases, and trained model artifacts.

## Setup

-   **Python Version:** Specified in `.python-version` (e.g., 3.8.x).
-   **Dependencies:** Install via `pip install -r requirements.txt`. Key libraries include `pandas`, `numpy`, `scikit-learn`, `fastf1`, `torch`, `xgboost`, and `tensorflow`.

## Key Workflows

### 1. Data Preparation
Build and clean datasets from raw FastF1 or Ergast sources:
```bash
python data/scripts/dataset_builder.py
```

### 2. Hyperparameter Tuning
Tune models using Optuna. Example for Stage 1 XGBoost:
```bash
python -m src.utils.tune_optuna --data_stage1 data/processed/output1.csv --task binary --model xgb --n_trials 100 --outdir runs/tuning
```

### 3. Final Model Training
Train the complete stacking ensemble and save frozen artifacts:
```bash
python -m src.training.train_final --data_stage1 data/processed/output1.csv --data_stage2 data/processed/output2.csv --run_name final_v1 --verbose
```

### 4. Evaluation
Evaluate the hierarchical ensemble performance:
```bash
python -m src.training.evaluate_hierarchical_ensemble --data_stage1 data/processed/output1.csv --data_stage2 data/processed/output2.csv
```

## Technical Details

-   **Race-Grouped CV:** Implemented in `src/data/data.py` via `make_race_group_folds`. It balances fold sizes and label distributions while strictly separating races.
-   **Holdout Races:** Specific race IDs (e.g., 53, 73, 24, 75, 2) are reserved for final case studies and excluded from training.
-   **Compound Mapping:** Canonical mapping for `HARD`, `MEDIUM`, `SOFT`, `INTERMEDIATE`, and `WET`.
-   **Sequential Modeling:** TCN and RNN models utilize sliding windows of lap-level data (typically 8-12 laps) to capture temporal trends in tire degradation and race progress.
