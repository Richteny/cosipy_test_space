import pandas as pd
import numpy as np
import pathlib
import xarray as xr

path = "/data/scratch/richteny/thesis/cosipy_test_space/data/output/LHS/"
albpath = "/data/scratch/richteny/Ren_21_Albedo/"
outpath = "/data/scratch/richteny/thesis/io/data/output/albedo_files/"

#Load albedo observations
albobs = xr.open_dataset(albpath+"HEF_processed_albedos.nc")

i=0
for fp in pathlib.Path(path).glob('*.nc'):
    #print(fp.stem)
    if i % 300 == 0:
        print(f"Processing file {i}/3000")
    ds = xr.open_dataset(fp).sel(time=albobs.time)[['ALBEDO','HGT','N_Points']]
    #create weighted mean 
    weights = ds['N_Points'] / np.sum(ds['N_Points'])
    weighted_field = (ds['ALBEDO'] * weights).sum(dim=['lat','lon'])
    
    weighted_field.to_netcdf(outpath+str(fp.name))
    i += 1
