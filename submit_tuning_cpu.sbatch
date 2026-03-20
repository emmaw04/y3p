#!/bin/bash
#SBATCH --partition=tiger
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

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
echo "Python: $(which python)"
python -c "import src; print('src import OK')"

echo "Task=$TASK Model=$MODEL DataFile=$DATAFILE NTrials=$NTRIALS"
echo "Running on $(hostname)"

if [ "$TASK" = "binary" ]; then
    python -m src.utils.tune_hyperparameters \
        --data_stage1 "$DATAFILE" \
        --task "$TASK" \
        --model "$MODEL" \
        --n_trials "$NTRIALS" \
        --outdir "runs/tuning/${TASK}_${MODEL}"
else
    python -m src.utils.tune_hyperparameters \
        --data_stage2 "$DATAFILE" \
        --task "$TASK" \
        --model "$MODEL" \
        --n_trials "$NTRIALS" \
        --outdir "runs/tuning/${TASK}_${MODEL}"
fi
