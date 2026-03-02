#!/bin/bash
#SBATCH --job-name=tune8
#SBATCH --partition=tiger
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/tune8/%x_%j.out
#SBATCH --error=slurm/tune8/%x_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

set -euo pipefail

# --- Init ---
source /etc/profile.d/modules.sh
module purge
# Use the system default Python (3.9.x on DCS tiger), since your venv was built for it.
# If you *do* have a Python module you want, load it here instead (e.g., Python/3.9.25).

# --- Project ---
cd /dcs/23/u5514578/cs310/y3p
source venv/bin/activate

# Reduce oversubscription (important for sklearn/XGBoost on multi-core)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Optional: keep TF CPU threads sensible (only matters for TCN/ANN jobs)
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=2

# --- Run ---
echo "Job started on $(hostname) at $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK  MEM: $SLURM_MEM_PER_NODE"
echo "CMD: $CMD"
eval "$CMD"
echo "Job finished at $(date)"
