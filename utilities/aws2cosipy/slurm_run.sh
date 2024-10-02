#!/bin/bash -l

#SBATCH --job-name="AWS2CO_3"
#SBATCH --qos=short
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=20
#SBATCH --ntasks=40
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy_test_space/utilities/aws2cosipy/
#SBATCH --account=morsanat
#SBATCH --error=slurm_run.err
##SBATCH --partition=computehm
#SBATCH --output=slurm_run.out
#SBATCH --mail-type=NONE


# The batch system should use the current directory as working directory.


#module load anaconda/2019.07 
conda activate horayzon_all

#python aws2cosipy.py -c ../../data/input/Abramov/Abramov_ERA5mod_spinup_Forcing_2009-2020.csv -o ../../data/input/Abramov/Abramov_300m_ERA5mod_spinup_2009-2020.nc -s ../../data/static/Abramov_300m_static_test.nc

#python aws2cosipy_saveonlyG.py -c ../../data/input/Abramov/Abramov_ERA5_1999_2021.csv -o ../../data/input/Abramov/Abramov_30m_SW_Moelg2009_2009_2010.nc -s ../../data/static/Abramov_30m_static_25kmhrzd.nc -b 20100101 -e 20101231
#python createLUT_crop.py -c ../../data/input/Abramov/Abramov_ERA5mod_spinup_Forcing_2009-2020.csv -o ../../data/input/Abramov/Abramov_30m_SW_Moelg_2014-2015.nc -s ../../data/static/Abramov_30m_static_25kmhrzd.nc -b 20140101 -e 20150101 
python cosmo2cosipy.py -c ../../data/input/HEF/COSMO_forcing_1999-2010.csv -o ../../data/input/HEF/HEF_COSMO_1D10m_HORAYZON_1999_2010.nc -s ../../data/static/HEF/HEF_static_10m_elevbands.nc -b 19990101 -e 20100101 #-sx ../../data/static/HEF/HEF_static_raw.nc 
