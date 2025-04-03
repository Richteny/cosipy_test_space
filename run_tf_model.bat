#!/bin/bash -l

#SBATCH --job-name="HEF-NN"
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
##SBATCH --ntasks=40
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy_test_space/
#SBATCH --account=morsanat
#SBATCH --error=TF_master.err
##SBATCH --reservation=PyMC
#SBATCH --partition=computehm
#SBATCH --output=TF_master.out
##SBATCH --mail-type=ALL

##ntasks between 10 and 20 
##chdir and python below must have current cosipy folder

echo $SLURM_CPUS_ON_NODE

conda activate tf_env
#module load anaconda/2019.07

python -u /data/scratch/richteny/thesis/cosipy_test_space/run_NN_dailyTSLA.py

