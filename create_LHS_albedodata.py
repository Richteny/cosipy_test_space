import pandas as pd
import numpy as np
import pathlib
import xarray as xr

#path = "/data/scratch/richteny/thesis/cosipy_test_space/data/output/LHS/"
path = "/data/scratch/richteny/thesis/io/data/output/bestfiles/"
albpath = "/data/scratch/richteny/Ren_21_Albedo/"
outpath = "/data/scratch/richteny/thesis/io/data/output/albedo_files/MCMC/"

#Load albedo observations
albobs = xr.open_dataset(albpath+"HEF_processed_HRZ-20CC-filter_albedos.nc")

i=0
for fp in pathlib.Path(path).glob('*.nc'):
    
    print(fp.stem)
    if i % 100 == 0:
        print(f"Processing file {i}/2500")
    #ds = xr.open_dataset(fp).sel(time=albobs.time)[['ALBEDO','HGT','N_Points']]
    ds = xr.open_dataset(fp)[['ALBEDO','HGT','MB','N_Points']]
    #create weighted mean 
    weights = ds['N_Points'] / np.sum(ds['N_Points'])
    weighted_alb = (ds['ALBEDO'] * weights).sum(dim=['lat','lon'])
    weighted_mb = (ds['MB'] * weights).sum(dim=['lat','lon'])

    # Combine into a new Dataset
    result = xr.Dataset({
        'ALBEDO_weighted': weighted_alb,
        'MB_weighted': weighted_mb
    })

    # Save to file
    #weighted_alb.to_netcdf(outpath + str(fp.name))
    result.to_netcdf(outpath + str(fp.name))
    i += 1
