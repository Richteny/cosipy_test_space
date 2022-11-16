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


#module load anaconda/2019.07 
conda activate cosipy_test
#python aws2cosipy.py -c ../../data/input/Abramov/Abramov_ERA5L_1981_2019.csv -o ../../data/input/Abramov/Abramov_ERA5L_1981_2019.nc -s ../../data/static/Abramov_static.nc

#python aws2cosipy_saveonlyG.py -c ../../data/input/Abramov/Abramov_ERA5_1999_2021.csv -o ../../data/input/Abramov/Abramov_30m_SW_Moelg2009_2009_2010.nc -s ../../data/static/Abramov_30m_static_25kmhrzd.nc -b 20100101 -e 20101231
python createLUT_crop.py -c ../../data/input/Abramov/Abramov_ERA5_1999_2021.csv -o ../../data/input/Abramov/Abramov_30m_SW_Wohlfahrt2016_2009-2010.nc -s ../../data/static/Abramov_30m_static_25kmhrzd.nc -b 20090101 -e 20100101 
