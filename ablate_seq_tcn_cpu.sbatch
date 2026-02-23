#!/bin/bash
#SBATCH --job-name=stage1_seq_tcn_cpu
#SBATCH --partition=tiger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3000
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

set -euo pipefail

cd /dcs/23/u5514578/cs310/y3p
source venv/bin/activate

# Make absolutely sure TF doesn't try GPU paths
export CUDA_VISIBLE_DEVICES=""
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTEROP_THREADS=1

# Keep caches off HOME
export XDG_CACHE_HOME="/tmp/$USER/xdgcache"
export KERAS_HOME="/tmp/$USER/keras"
export TMPDIR="/tmp/$USER/tmp"
mkdir -p "$XDG_CACHE_HOME" "$KERAS_HOME" "$TMPDIR"

echo "Host: $(hostname)"
python -c "import tensorflow as tf; print('TF', tf.__version__, 'GPUs', tf.config.list_physical_devices('GPU'))"

python3 -m src.utils.ablation_eval \
  --data_stage1 data/processed/output1.csv \
  --outdir runs/ablation \
  --run_name stage1_seq_tcn_ablation_CPU \
  --mode seq \
  --seq_model tcn \
  --n_splits 5 \
  --seed 42 \
  --seq_len 8 \
  --pad_left \
  --add_timestep_mask
