#!/bin/bash -l

#SBATCH --job-name="emulator_mcmc"
#SBATCH --output=logs/mcmc_%A_%a.out
#SBATCH --qos=medium
#SBATCH --error=logs/mcmc_%A_%a.err
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy_test_space/
#SBATCH --account=morsanat
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
##SBATCH --time=12:00:00
##SBATCH --mem=16G

conda activate pymc_env

#python -u emulator_test_mcmc.py ${SLURM_ARRAY_TASK_ID}
#python -u point_emulator_test_mcmc.py ${SLURM_ARRAY_TASK_ID}
#python -u emulator_testsyserr_mcmc.py ${SLURM_ARRAY_TASK_ID}
#python -u emulator_firststage_mcmc.py ${SLURM_ARRAY_TASK_ID}
#python -u emulator_secondstage_mcmc.py ${SLURM_ARRAY_TASK_ID}
python -u halji_emul.py ${SLURM_ARRAY_TASK_ID}
