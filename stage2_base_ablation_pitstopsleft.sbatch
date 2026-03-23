#!/bin/bash
#SBATCH --job-name=base-ablation-s2
#SBATCH --partition=tiger
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=/dcs/23/u5514578/cs310/y3p/logs/base_ablation_s2_%j.out
#SBATCH --error=/dcs/23/u5514578/cs310/y3p/logs/base_ablation_s2_%j.err

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
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "Python: $(which python)"
python -V
python -c "import src.utils.base_ablation_eval; print('base_ablation_eval import OK')"

python -m src.utils.base_ablation_eval \
  --data_path data/processed/pit_stops_left_dataset/dataset2_pitstopsleft.csv \
  --stage 2 \
  --models "rf,xgb,svm,tcn_gru" \
  --outdir runs/ablation \
  --run_name stage2_base_ablation_pitstopsleft \
  --n_splits 5 \
  --seed 42 \
  --verbose
