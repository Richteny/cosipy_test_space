#!/bin/bash

#SBATCH --job-name="MaEraAbr"
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy_test_space/
#SBATCH --account=morsanat
#SBATCH --error=Control_master.err
#SBATCH --partition=computehm
#SBATCH --output=Control_master.out
#SBATCH --mail-type=ALL

##ntasks between 10 and 20 
##chdir and python below must have current cosipy folder

echo $SLURM_CPUS_ON_NODE

module load anaconda/2019.07

#python -u /data/scratch/richteny/thesis/cosipy_test_space/run_spotpy_full.py #COSIPY.py
python -u /data/scratch/richteny/thesis/cosipy_test_space/COSIPY.py

