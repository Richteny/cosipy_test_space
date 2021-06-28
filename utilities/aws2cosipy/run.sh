#!/bin/bash -l

# The batch system should use the current directory as working directory.


module load anaconda/2019.07 

python3 aws2cosipy.py -c ../../data/input/Abramov/Abramov_ERA5L_1981_2019.csv -o ../../data/input/Abramov/Abramov_ERA5L_1981_2016.nc -s ../../data/static/Abramov_static.nc -b 19810101 -e 20191231 
