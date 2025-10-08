import xarray as xr
import numpy as np
import pathlib
import datetime
import pandas as pd

datapath="/data/scratch/richteny/thesis/cosipy_test_space/data/output/LHS/"
outpath="/data/scratch/richteny/thesis/io/data/output/reduced_bestfiles/"

def spatial_mean(ds):
    weights = ds['N_Points'] / np.sum(ds['N_Points'])
    result = ds.copy()
    for var in list(ds.variables):
        if var not in ["time", "lat", "lon", "HGT", "MASK", "SLOPE", "ASPECT", "N_Points"]:
            weighted_field = (ds[var] * weights).sum(dim=['lat', 'lon'])
            result[var] = weighted_field
    return result

#def get_closest_elevgridcell(ds, elevation_target=3048.0):
#
#    #Recorded original coordinates could not be converted into lat/lon coordinates due to un$
#    #Reconstructed positions from later recorded GPS points of AWS locations (+/-100 m):
#    #HEF lower: 46.813570° N; 10.788977° E; 2640 m
#    #HEF upper: 46.790453° N; 10.747121° E; 3048 m
#    abs_diff = abs(ds['HGT'] - elevation_target)
#    closest_cells = abs_diff.where(abs_diff == abs_diff.min(), drop=True)
#    lat = closest_cells.lat
#    lon = closest_cells.lon
#    return ds.sel(lat=lat, lon=lon)

start = datetime.datetime.now()

#location = "upper"
#dic_elev = {'upper': 3048.0,
#            'lower': 2640.0}

i = 0
for fp in pathlib.Path(datapath).glob('HEF_COSMO_1D20m_1999_2010_HORAYZON_IntpPRES*.nc'):
    #print(fp)
    if i%50 == 0:
        print(f"Processing file {i}/3000")
    ds = xr.open_dataset(fp)
    #for cumulative mass balances create timeseries of MB as dataframe?
    #spat_mean = ds[['MB']].sel(time=slice("2000-01-01","2009-12-31")).mean(dim=['lat','lon'])
    #ds = ds[['G','ALBEDO','LWin','LWout','ME','H','LE','B','QRR','HGT','N_Points','TS','MB']].sel(time=slice("2000-01-01","2009-12-31"))
    #sub = get_closest_elevgridcell(ds, elevation_target=dic_elev[location]).isel(lat=0,lon=0)
    #let's start with the basic ones first
    ds = ds[['ALBEDO','HGT','N_Points','TS','MB']] #.sel(time=slice("2000-01-01","2009-12-31"))
    sub = spatial_mean(ds)
    sub = sub[['ALBEDO','TS','MB']]
    
    sub.to_netcdf(outpath+f'spatialmeans_cali_'+str(fp.name))
    i+=1

print("Done after", datetime.datetime.now()-start)

