import xarray as xr
import numpy as np
import pathlib

path = "/data/scratch/richteny/thesis/io/data/output/bestfiles/"
outpath = "/data/scratch/richteny/thesis/io/data/output/aws_eb/"

def get_closest_elevgridcell(ds, elevation_target=3048.0):
    abs_diff = abs(ds['HGT'] - elevation_target)
    closest_cells = abs_diff.where(abs_diff == abs_diff.min(), drop=True)
    lat = closest_cells.lat
    lon = closest_cells.lon
    return ds.sel(lat=lat, lon=lon)

for fp in pathlib.Path(path).glob('*.nc'):
    print(fp.stem)
    #raw_fp = str(fp.stem).split('HEF_COSMO_1D20m_1999_2010_HORAYZON_IntpPRES_MCMC-ensemble_19990101-20091231_RRR-')[-1]
    #rrr_factor = float(raw_fp.split('_')[0])
    #alb_snow = float(raw_fp.split('_')[1])
    #alb_ice = float(raw_fp.split('_')[2])
    #alb_firn = float(raw_fp.split('_')[3])
    #alb_aging = float(raw_fp.split('_')[4])
    #alb_depth = float(raw_fp.split('_')[5])
    #roughness_fresh_snow = float(raw_fp.split('_')[6])
    #roughness_ice = float(raw_fp.split('_')[7])
    #roughness_firn = float(raw_fp.split('_')[8])
    #aging_factor_roughness = float(raw_fp.split('_')[9])
    #key = f"{rrr_factor}_{alb_snow}_{alb_ice}_{alb_firn}_{alb_aging}_{alb_depth}_{roughness_ice}"

    ds = xr.open_dataset(fp)
    sub = get_closest_elevgridcell(ds).isel(lat=0,lon=0)
    sub = sub.sel(time=slice("2003-10-01",None))
    sub['SWnet'] = sub['G'] * (1 - sub['ALBEDO'])
    df = sub.to_dataframe()
    df.to_csv(outpath+str(fp.stem)+'.csv')
