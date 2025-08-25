import pandas as pd
import numpy as np
import pathlib
import xarray as xr

path = "/data/scratch/richteny/thesis/cosipy_test_space/data/output/manual_sensitivity_test/"
#path = "/data/scratch/richteny/thesis/io/data/output/bestfiles/"
albpath = "/data/scratch/richteny/Ren_21_Albedo/"
outpath = "/data/scratch/richteny/thesis/io/data/output/albedo_files/sens_test/"
outpath_me = "/data/scratch/richteny/thesis/io/data/output/sens_me/"

#Load albedo observations
albobs = xr.open_dataset(albpath+"HEF_processed_HRZ-30CC-filter_albedos.nc")

i=0
for fp in pathlib.Path(path).glob('*.nc'):
    
    print(fp.stem)
    if i % 100 == 0:
        print(f"Processing file {i}/2500")
    #ds = xr.open_dataset(fp).sel(time=albobs.time)[['ALBEDO','HGT','N_Points']]
    ds = xr.open_dataset(fp)[['ALBEDO','HGT','MB','N_Points','ME']]
    #create weighted mean 
    weights = ds['N_Points'] / np.sum(ds['N_Points'])
    weighted_alb = (ds['ALBEDO'] * weights).sum(dim=['lat','lon'])
    weighted_mb = (ds['MB'] * weights).sum(dim=['lat','lon'])
    weighted_melt = (ds[['ME']] * weights).sum(dim=['lat','lon'])

    #melt per season
    time = weighted_melt['time']
    hydroyear = time.dt.year.where(time.dt.month>=10, time.dt.year -1)
    ds_w = weighted_melt.assign_coords(hydroyear=hydroyear)

    mask_ablation = time.dt.month.isin([5,6,7,8,9])

    ds_abl = (ds_w.where(mask_ablation).groupby('hydroyear').mean(dim='time'))
    mask_hyear = ((time.dt.month >= 10) | (time.dt.month <= 9))
    ds_annual = ds_w.groupby('hydroyear').mean(dim='time')
    #rename so merge is possible
    ds_abl_rename = ds_abl.rename({'ME': 'ME_abl'})
    ds_annual_rename = ds_annual.rename({'ME': 'ME_ann'})
    ds_merge = xr.merge([ds_abl_rename, ds_annual_rename])

    # Combine into a new Dataset
    result = xr.Dataset({
        'ALBEDO_weighted': weighted_alb,
        'MB_weighted': weighted_mb
    })

    # Save to file
    #weighted_alb.to_netcdf(outpath + str(fp.name))
    result.to_netcdf(outpath + str(fp.name))
    ds_merge.to_netcdf(outpath_me + str(fp.name))
    i += 1
