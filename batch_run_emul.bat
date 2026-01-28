#!/bin/bash -l

#SBATCH --job-name="emulator_mcmc"
##SBATCH --output=logs/mcmc_%A_%a.out
#SBATCH --output=logs/mcmc.out
#SBATCH --error=logs/mcmc.err
#SBATCH --qos=medium
##SBATCH --error=logs/mcmc_%A_%a.err
##SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy_test_space/
#SBATCH --account=morsanat
#SBATCH --partition=compute
#SBATCH --nodes=1
##SBATCH --time=12:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=20

source /data/scratch/richteny/miniconda3/etc/profile.d/conda.sh

conda activate tf_env

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=-1

export TF_NUM_INTEROP_THREADS=20
export TF_NUM_INTRAOP_THREADS=20
#conda activate pymc_env

#python -u emulator_test_mcmc.py ${SLURM_ARRAY_TASK_ID}
#python -u point_emulator_test_mcmc.py ${SLURM_ARRAY_TASK_ID}
#python -u emulator_testsyserr_mcmc.py ${SLURM_ARRAY_TASK_ID}
#python -u emulator_firststage_mcmc.py ${SLURM_ARRAY_TASK_ID}
#python -u emulator_secondstage_mcmc.py ${SLURM_ARRAY_TASK_ID}
#python -u halji_emul.py ${SLURM_ARRAY_TASK_ID}
python -u tfp_calibration_nuts.py
