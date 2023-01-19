#!/bin/bash -l

#SBATCH --job-name="AWS2CO_1"
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy_test_space/utilities/aws2cosipy/
#SBATCH --account=morsanat
#SBATCH --error=crop_run.err
#SBATCH --partition=computehm
#SBATCH --output=crop_run.out
#SBATCH --mail-type=NONE


# The batch system should use the current directory as working directory.


module load anaconda/2019.07 
#conda activate cosipy_test
python aws2cosipy.py -c ../../data/input/Abramov/Abramov_ERA5mod_spinup_Forcing_2009-2020.csv -o ../../data/input/Abramov/Abramov_1D30m_ERA5mod_spinup_Wohlfahrt_2009-2020.nc -s ../../data/static/Abramov_hrzd1D_30m_elev.nc

#python crop_file_to_glacier.py -i ../../data/input/Abramov/Abramov_300m_ERA5mod_spinup_Wohlfahrt_2009-2020.nc -o ../../data/input/Abramov/Abramov_300m_ERA5mod_spinup_Wohlfahrt_2009-2020_crop.nc
