# Data-driven forecasting of optimal pit stop strategies in Formula 1

This repository contains the code for my third-year project on predicting Formula 1 pit stop strategy from historical race data.

The project focuses on the online version of the problem: at the end of each lap, given the current race situation, should a driver pit now or stay out, and if they pit, which compound should they switch to?

Rather than trying to predict a full race strategy in one go, the system works lap by lap. This makes it much closer to how strategy decisions actually happen during a race.

## What the model does

The final system is split into two stages.

### Stage 1: pit or no pit

A binary classifier predicts the probability that a driver should pit at the end of the current lap.

### Stage 2: next compound

If Stage 1 predicts a pit stop, a second model predicts the tyre compound for the next stint.

The overall setup uses stacked ensembling. Different base learners make probability predictions, and a meta-learner combines them into the final output.

## Project aim

The goal is not just to copy historical pit stops, but to build a model that learns useful strategic patterns from race context, such as:

- tyre age and current compound
- race progress
- track position and gaps to nearby cars
- traffic and possible undercut or overcut situations
- safety car and virtual safety car phases
- weather and wet or dry transitions
- estimated rejoin gaps after a pit stop

A lot of previous work in this area only looked at dry races or used evaluation setups that made the task easier than it would be in reality. This project tries to be stricter about that by using race-wise splits and holding out entire races for final case studies.

## Data

The data mainly comes from the FastF1 API, covering seasons from 2018 to 2025.

FastF1 was used because it provides much richer session data than basic race result tables, including lap timing, tyre information, weather, track status, and race control data. The raw session data is downloaded and stored in a SQLite database, then cleaned and turned into modelling datasets.

There are two main processed datasets:

- `dataset1.csv` for Stage 1 pit-stop prediction
- `dataset2.csv` for Stage 2 compound prediction

The Stage 1 dataset is lap-level and highly imbalanced, since most laps are non-pit laps.  
The Stage 2 dataset only includes pit events, since compound choice only matters when a stop happens.

## Modelling approach

A range of tabular and sequential models were tested during development.

These include:

- XGBoost
- Random Forest
- SVM
- feed-forward neural networks
- sequential deep learning models such as TCN and recurrent variants

The final system uses stacking, where base model probabilities are fed into an XGBoost meta-learner.

For Stage 1, the main focus is strong rare-event prediction and good ranking of pit opportunities.  
For Stage 2, the focus is choosing the most likely next compound once a stop is predicted.

## Validation

The project uses **5-fold race-wise cross-validation**, grouped by `race_id`.

This means laps from the same race are never split across training and validation folds. That is important because random lap-level splitting leaks race-specific context and can make results look much better than they really are.

Final evaluation is done on held-out races that are kept separate from model development.

## Repository layout

### `src/`

Main modelling and training code.

- `src/data/`  
  Dataset schemas, fold handling, and sequence-building logic
- `src/models/`  
  Model definitions and factories for the different learners
- `src/training/`  
  Scripts for base model evaluation, stacking, and final training
- `src/utils/`  
  Utility scripts for hyperparameter tuning, threshold tuning, holdout evaluation, SHAP analysis, and ablation studies

### `fastf1/`

Data collection and database building.

- `download_f1_data.py` downloads session data
- `build_fastf1_db.py` builds the SQLite database
- `db_clean.py` applies cleaning steps
- `dataset_builder.py` creates the final modelling datasets

### `data/`

Stored data.

- `data/raw/` contains the SQLite databases
- `data/processed/` contains the processed CSV datasets and related outputs

### `runs/`

Saved experiment outputs.

### VSE
copied from heilmeiers directory.
code changes were made in VSE/racesim/

This includes:

- tuned hyperparameters
- trained model artifacts
- holdout predictions and metrics
- SHAP outputs
- ablation results
- confusion matrices
- SMOTE experiments for Stage 2

## Typical workflow

### 1. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build the processed datasets

```bash
python fastf1/dataset_builder.py
```

### 3. Tune a model

Example for Stage 1 XGBoost:

```bash
python -m src.utils.tune_hyperparameters \
    --data_stage1 data/processed/dataset1.csv \
    --task binary \
    --model xgb \
    --n_trials 100 \
    --outdir runs/tuning
```

### 4. Train the final pipeline

```bash
python -m src.training.train_final \
    --data_stage1 data/processed/dataset1.csv \
    --data_stage2 data/processed/dataset2.csv \
    --run_name final_run \
    --verbose
```

### 5. Run holdout evaluation and interpretation

```bash
python -m src.utils.evaluate_holdout \
    --data_stage1 data/processed/dataset1.csv \
    --data_stage2 data/processed/dataset2.csv
```

```bash
python -m src.utils.SHAP_eval --model_path runs/final_run/stage1_binary/artifacts/...
```

## Metrics

Because pit stops are rare, standard accuracy is not very useful for Stage 1.

### Stage 1

Main metrics are:

- PR-AUC
- F1-score
- precision and recall
- log loss or calibration-focused metrics where relevant

### Stage 2

Main metrics are:

- macro F1
- balanced accuracy
- top-k style accuracy where useful

## Notes

A few extra things in the repository are there for analysis rather than the final deployed pipeline, including:

- SHAP-based interpretation
- base and meta ablation studies
- SMOTE experiments for Stage 2
- holdout race case studies
- comparison against race simulation outputs

## Overall

The point of this project is to build a pit-stop decision system that works from the information available up to the current lap, rather than using future information or simplified offline assumptions.

So the repository is really a mix of:

- data engineering
- feature engineering
- imbalanced classification
- sequential modelling
- ensemble learning
- evaluation in a motorsport strategy setting