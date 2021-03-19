#!/bin/bash -l

#SBATCH --job-name="AWS2CO"
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy/utilities/aws2cosipy/
#SBATCH --account=morsanat
#SBATCH --error=slurm_run.err
##SBATCH --partition=computehm
#SBATCH --output=slurm_run.out
#SBATCH --mail-type=NONE


# The batch system should use the current directory as working directory.


module load anaconda/2019.07 

#python aws2cosipy.py -c ../../data/input/Abramov/Abramov_ERA5L_1981_2019.csv -o ../../data/input/Abramov/Abramov_ERA5L_1981_2019.nc -s ../../data/static/Abramov_static.nc

python aws2cosipy.py -c ../../data/input/Abramov/Abramov_ERA5L_fix_1982_2019.csv -o ../../data/input/Abramov/Abramov_180m_ERA5L_fix_2000_2019.nc -s ../../data/static/Abramov_180m_static.nc -b 20000101 -e 20191231
