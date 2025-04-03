import xarray as xr
import numpy as np
import pathlib

path = "/data/scratch/richteny/thesis/cosipy_test_space/data/output/"

for fp in pathlib.Path(path).glob('HEF_COSMO_*.nc'):
    print(fp)
    ds = xr.open_dataset(fp)
    fname = "avg_glacierwide_" + str(fp.stem) + '.csv'
    if "N_Points" in list(ds.keys()):
        dsmb = ds['MB'] * ds['N_Points'] / np.sum(ds['N_Points'])
        spatial_mean = dsmb.sum(dim=['lat','lon'])
        dfmb = spatial_mean.to_dataframe(name="MB")
    else:
        spatial_mean = ds.MB.mean(dim=['lat','lon'])
        dfmb = spatial_mean.to_dataframe("MB")
    dfmb.to_csv(path+fname)
