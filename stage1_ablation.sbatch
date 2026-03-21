#!/bin/bash
#SBATCH --job-name=stage1-ablation
#SBATCH --partition=tiger
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=/dcs/23/u5514578/cs310/y3p/logs/stage1_ablation_%j.out
#SBATCH --error=/dcs/23/u5514578/cs310/y3p/logs/stage1_ablation_%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

REPO_DIR="/dcs/23/u5514578/cs310/y3p"
cd "$REPO_DIR" || exit 1

source venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job: $SLURM_JOB_NAME"
echo "Repo: $REPO_DIR"
echo "PWD: $(pwd)"
echo "Host: $(hostname)"
echo "Python: $(which python)"
python -V
python -c "import src; print('src import OK')"

echo "Checking dataset exists..."
ls -lh data/processed/dataset1.csv

echo "Running stage 1 ablation..."
python -m src.utils.ablation_eval \
  --stage 1 \
  --data_path data/processed/dataset1.csv \
  --outdir runs/ablation \
  --run_name stage1_meta_ablation \
  --n_splits 5 \
  --seed 42 \
  --verbose