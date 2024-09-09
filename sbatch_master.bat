#!/bin/bash -l

#SBATCH --job-name="HEF-Test"
#SBATCH --qos=medium
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=20
#SBATCH --ntasks=40
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy_test_space-v2/
#SBATCH --account=morsanat
#SBATCH --error=Control_master.err
##SBATCH --partition=computehm
#SBATCH --output=Control_master.out
##SBATCH --mail-type=ALL

##ntasks between 10 and 20 
##chdir and python below must have current cosipy folder

echo $SLURM_CPUS_ON_NODE

#conda activate cspy
#conda activate pymc3_env
conda activate pymc_env
#module load anaconda/2019.07

python -u /data/scratch/richteny/thesis/cosipy_test_space-v2/COSIPY.py
#python -u /data/scratch/richteny/thesis/cosipy_test_space-v2/pymc_calibration.py
#python -u /data/scratch/richteny/thesis/cosipy_test_space-v2/EMERGENCY_pymc_calibration.py

