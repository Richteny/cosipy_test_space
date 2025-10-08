#!/bin/bash

#SBATCH --job-name="MaEraAbr"
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy/
#SBATCH --account=morsanat
#SBATCH --error=Control_master.err
##SBATCH --partition=computehm
#SBATCH --output=Control_master.out
#SBATCH --mail-type=ALL

##ntasks between 10 and 20 
##chdir and python below must have current cosipy folder

echo $SLURM_CPUS_ON_NODE

module load anaconda/2019.07

python -u /data/scratch/richteny/thesis/cosipy/COSIPY.py


