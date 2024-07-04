#!/bin/bash -l

# The batch system should use the current directory as working directory.
#SBATCH --job-name=REAL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=03:00:00

module load intel64 netcdf 

export KMP_STACKSIZE=64000000
export OMP_NUM_THREADS=1
ulimit -s unlimited

python3 aws2cosipy.py -c ../../data/input/Abramov/Abramov_ERA5L_1981_2019.csv -o ../../data/input/Abramov/Abramov_ERA5L_1981_2016.nc -s ../../data/static/Abramov_static.nc -b 19810101 -e 20191231 
