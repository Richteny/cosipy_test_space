#!/bin/bash -l

#SBATCH --job-name="AWS2CO_3"
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy_test_space/cosipy/utilities/aws2cosipy/
#SBATCH --account=morsanat
#SBATCH --error=slurm_run.err
#SBATCH --partition=compute
#SBATCH --output=slurm_run.out
#SBATCH --mail-type=NONE


# The batch system should use the current directory as working directory.


#module load anaconda/2019.07 
conda activate horayzon_all

#python aws2cosipy.py -c ../../data/input/Abramov/Abramov_ERA5mod_spinup_Forcing_2009-2020.csv -o ../../data/input/Abramov/Abramov_300m_ERA5mod_spinup_2009-2020.nc -s ../../data/static/Abramov_300m_static_test.nc

#python aws2cosipy_saveonlyG.py -c ../../data/input/Abramov/Abramov_ERA5_1999_2021.csv -o ../../data/input/Abramov/Abramov_30m_SW_Moelg2009_2009_2010.nc -s ../../data/static/Abramov_30m_static_25kmhrzd.nc -b 20100101 -e 20101231
#python createLUT_crop.py -c ../../data/input/Abramov/Abramov_ERA5mod_spinup_Forcing_2009-2020.csv -o ../../data/input/Abramov/Abramov_30m_SW_Moelg_2014-2015.nc -s ../../data/static/Abramov_30m_static_25kmhrzd.nc -b 20140101 -e 20150101 
python cosmo2cosipy.py -i "../../data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ../../data/input/HEF/HEF_COSMO_HORAYZON_1999_2010_.nc -s ../../data/static/HEF/HEF_static_20m_elevbands.nc -b 19990101 -e 20100101 --sw ../../data/static/HEF/LUT_HORAYZON_sw_dir_cor_1D20m.nc 
