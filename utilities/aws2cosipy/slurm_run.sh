#!/bin/bash -l

#SBATCH --job-name="AWS2CO_3"
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy_test_space/utilities/aws2cosipy/
#SBATCH --account=morsanat
#SBATCH --error=slurm_run.err
#SBATCH --partition=computehm
#SBATCH --output=slurm_run.out
#SBATCH --mail-type=NONE


# The batch system should use the current directory as working directory.


module load anaconda/2019.07 

#python aws2cosipy.py -c ../../data/input/Abramov/Abramov_ERA5L_1981_2019.csv -o ../../data/input/Abramov/Abramov_ERA5L_1981_2019.nc -s ../../data/static/Abramov_static.nc

python aws2cosipy.py -c ../../data/input/Abramov/Abramov_ERA5L_fix_1982_2019.csv -o ../../data/input/Abramov/Abramov_1350m_ERA5L_duplicate_fix_lr_prcp_2013_2017.nc -s ../../data/static/Abramov_1350m_static.nc -b 20121001 -e 20170930
