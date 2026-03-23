#!/bin/bash
#SBATCH --job-name=train-final-s1
#SBATCH --partition=tiger
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=/dcs/23/u5514578/cs310/y3p/logs/train_final_stage1_%j.out
#SBATCH --error=/dcs/23/u5514578/cs310/y3p/logs/train_final_stage1_%j.err

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

python -c "import src; print('src import OK')"
python -c "import src.training.train_final; print('train_final import OK')"

ls -lh data/processed/dataset1.csv

python -m src.training.train_final \
  --data_stage1 data/processed/dataset1.csv \
  --only_stage stage1
