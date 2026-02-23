#!/bin/bash
#SBATCH --job-name=ablate_base_svm_xgb
#SBATCH --partition=tiger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3000
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

set -euo pipefail

cd /dcs/23/u5514578/cs310/y3p
source venv/bin/activate

export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python3 -m src.utils.ablation_eval \
  --data_stage1 data/processed/output1.csv \
  --mode base \
  --base_models svm,xgb \
  --n_splits 5 \
  --seed 42
