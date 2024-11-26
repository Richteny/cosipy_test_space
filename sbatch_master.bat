#!/bin/bash -l

#SBATCH --job-name="HEF-Sens"
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
##SBATCH --ntasks=40
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy_test_space/
#SBATCH --account=morsanat
#SBATCH --error=Control_master.err
#SBATCH --reservation=PyMC
#SBATCH --partition=compute
#SBATCH --output=Control_master.out
##SBATCH --mail-type=ALL

##ntasks between 10 and 20 
##chdir and python below must have current cosipy folder

echo $SLURM_CPUS_ON_NODE

#conda activate cspy
#conda activate pymc3_env
conda activate pymc_env
#module load anaconda/2019.07

#python -u /data/scratch/richteny/thesis/cosipy_test_space/COSIPY.py
#python -u /data/scratch/richteny/thesis/cosipy_test_space/FAST_spotpy.py
#python -u /data/scratch/richteny/thesis/cosipy_test_space/pymc_calibration.py
#python -u /data/scratch/richteny/thesis/cosipy_test_space/synthetic_pymc_calibration.py
python -u /data/scratch/richteny/thesis/cosipy_test_space/LHS_synth_surrogate_params.py
#python -u /data/scratch/richteny/thesis/cosipy_test_space/EMERGENCY_pymc_calibration.py

