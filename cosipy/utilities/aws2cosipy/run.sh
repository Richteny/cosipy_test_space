#!/bin/bash -l

# The batch system should use the current directory as working directory.


module load anaconda/2019.07 

python3 aws2cosipy.py -i ../../data/input/Zhadang/Zhadang_ERA5_200901_short.csv -o ../../data/input/Zhadang/Zhadang_ERA5_2009.nc -s ../../data/static/Zhadang_static.nc -b 20090101 -e 20091231 
