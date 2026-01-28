#!/bin/bash -l

#SBATCH --job-name="AWS2CO_3"
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy_test_space/cosipy/utilities/aws2cosipy/
#SBATCH --account=morsanat
#SBATCH --error=slurm_run.err
#SBATCH --partition=computehm
#SBATCH --output=slurm_run.out
#SBATCH --mail-type=NONE


# The batch system should use the current directory as working directory.


#module load anaconda/2019.07 
conda activate horayzon_all
cd /data/scratch/richteny/thesis/cosipy_test_space/
#python aws2cosipy.py -c ../../data/input/Abramov/Abramov_ERA5mod_spinup_Forcing_2009-2020.csv -o ../../data/input/Abramov/Abramov_300m_ERA5mod_spinup_2009-2020.nc -s ../../data/static/Abramov_300m_static_test.nc

#python aws2cosipy_saveonlyG.py -c ../../data/input/Abramov/Abramov_ERA5_1999_2021.csv -o ../../data/input/Abramov/Abramov_30m_SW_Moelg2009_2009_2010.nc -s ../../data/static/Abramov_30m_static_25kmhrzd.nc -b 20100101 -e 20101231
#python createLUT_crop.py -c ../../data/input/Abramov/Abramov_ERA5mod_spinup_Forcing_2009-2020.csv -o ../../data/input/Abramov/Abramov_30m_SW_Moelg_2014-2015.nc -s ../../data/static/Abramov_30m_static_25kmhrzd.nc -b 20140101 -e 20150101 
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_1500m_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_1500m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_1500m.nc 
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_1200m_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_1200m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_1200m.nc
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_1000m_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_1000m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_1000m.nc
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_900m_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_900m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_900m.nc
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_750m_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_750m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_750m.nc
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_600m_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_600m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_600m.nc
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_450m_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_450m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_450m.nc
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_300m_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_300m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_300m.nc
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_210m_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_210m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_210m.nc
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_150m_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_150m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_150m.nc
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_90m_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_90m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_90m.nc
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_60m_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_60m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_60m.nc
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_30m_mean_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_30m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_mean_30m.nc

# 1D below

#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_1D10m_mean_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_1D10m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_mean_1D10m.nc
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_1D20m_HORAYZON_1999_2010_IntpPRES_SFadj2.nc -s ./data/static/HEF/HEF_static_1D20m_new.nc -b 19990101 -e 20100101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_1D20m.nc
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_1D30m_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_1D30m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_1D30m.nc
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_1D50m_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_1D50m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_1D50m.nc
#python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/HEF/COSMO_forcing_1999-2010_PRESintp.csv" -o ./data/input/HEF/HEF_COSMO_1D100m_HORAYZON_1999_2001_IntpPRES_nosfcor.nc -s ./data/static/HEF/HEF_static_1D100m_new.nc -b 19990101 -e 20010101 --sw ./data/static/HEF/HEF_HORAYZON-LUT_1D100m.nc

python ./cosipy/utilities/aws2cosipy/cosmo2cosipy.py -i "./data/input/Halji/COSIPY_HARv2_1980_2022_RGI60-15.06065.csv" -o ./data/input/Halji/Halji_HARv2_1D20m_HORAYZON_1988_2022.nc -s ./data/static/Halji/Halji_combined_1D20m_SRF.nc -b 1987-10-01 -e  2022-12-31 --sw ./data/static/Halji/Halji_HORAYZON-LUT_1D20m.nc ./data/static/Halji/Halji2014_HORAYZON-LUT_1D20m.nc ./data/static/Halji/Halji2018_HORAYZON-LUT_1D20m.nc ./data/static/Halji/Halji2021_HORAYZON-LUT_1D20m.nc

# Point scale
#python ./cosipy/utilities/aws2cosipy/aws2cosipy.py -i "./data/input/HEF/cosipy_forcing_upper_station.csv" -o ./data/input/HEF/HEF_AWSU_2002-2004_RFpr.nc -s ./data/static/HEF/HEF_static_30m_AWSpoint.nc -b 20020901 -e 20040930

