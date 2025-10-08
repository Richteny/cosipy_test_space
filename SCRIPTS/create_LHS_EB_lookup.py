import pandas as pd
import numpy as np
import pathlib
import xarray as xr

path = "/data/scratch/richteny/thesis/cosipy_test_space/data/output/"
path += "LHS/"

def get_season(month):
    if month in [10,11,12,1,2,3,4]: #Accumulation (Oct 1 to Apr 30)
        return "Acc"
    elif month in [5,6,7,8,9]: #Ablation (may 1 to sep 30)
        return "Abl"
    else:
        print("Month not recognized.")
        return "Error"

def sum_eb_season(ds, var):
    weights = ds['N_Points'] / np.sum(ds['N_Points'])
    if var == "TS":
         ds[var] = ds[var] - 273.15
    weighted_melt_energy = (ds[var] * weights).sum(dim=['lat', 'lon'])
    weighted_melt_energy['season'] = ds['season']
    if var == "ME":
        seasonal_melt_energy = weighted_melt_energy.groupby('season').sum(dim='time') * (3600 / 1e6)  # Convert to MJ/m²
    elif var == "TS":
        seasonal_melt_energy = weighted_melt_energy.groupby('season').sum(dim='time')  
    else:
        seasonal_melt_energy = weighted_melt_energy.groupby('season').sum(dim='time') / (3600 / 1e6) #same for latent heat flux? Can still do afterwards
    return seasonal_melt_energy

results_list = []

for fp in pathlib.Path(path).glob('*.nc'):
    ds = xr.open_dataset(fp).sel(time=slice("2000-10-01","2009-09-30"))[['N_Points','LE','H','B','ME','TS']]
    ds['month']= ds.time.dt.month
    ds['season'] = xr.apply_ufunc(get_season, ds['month'], vectorize=True)
    
    acc_meltenergy = float(sum_eb_season(ds, "ME").sel(season="Acc"))
    abl_meltenergy = float(sum_eb_season(ds, "ME").sel(season="Abl"))
    acc_latent = float(sum_eb_season(ds, "LE").sel(season="Acc"))
    abl_latent = float(sum_eb_season(ds, "LE").sel(season="Abl"))
    acc_sensible = float(sum_eb_season(ds, "H").sel(season="Acc"))
    abl_sensible = float(sum_eb_season(ds, "H").sel(season="Abl"))
    acc_ground = float(sum_eb_season(ds, "B").sel(season="Acc"))
    abl_ground = float(sum_eb_season(ds, "B").sel(season="Abl"))
    acc_ts = float(sum_eb_season(ds, "TS").sel(season="Acc"))
    abl_ts = float(sum_eb_season(ds, "TS").sel(season="Abl"))

    print(fp.stem)
    raw_fp = str(fp.stem).split('HEF_COSMO_1D20m_1999_2010_HORAYZON_IntpPRES_LHS-narrow_19990101-20091231_RRR-')[-1]
    rrr_factor = float(raw_fp.split('_')[0])
    alb_snow = float(raw_fp.split('_')[1])
    alb_ice = float(raw_fp.split('_')[2])
    alb_firn = float(raw_fp.split('_')[3])
    alb_aging = float(raw_fp.split('_')[4])
    alb_depth = float(raw_fp.split('_')[5])
    roughness_fresh_snow = float(raw_fp.split('_')[6])
    roughness_ice = float(raw_fp.split('_')[7])
    roughness_firn = float(raw_fp.split('_')[8])
    aging_factor_roughness = float(raw_fp.split('_')[9])

    result = {'rrr_factor': rrr_factor,
              'alb_ice': alb_ice,
              'alb_snow': alb_snow,
              'alb_firn': alb_firn,
              'albedo_aging': alb_aging,
              'albedo_depth': alb_depth,
              'roughness_fresh_snow': roughness_fresh_snow,
              'roughness_ice': roughness_ice,
              'roughness_firn': roughness_firn,
              'aging_factor_roughness': aging_factor_roughness,
              'ME_acc': acc_meltenergy,
              'ME_abl': abl_meltenergy,
              'LE_acc': acc_latent,
              'LE_abl': abl_latent,
              'H_acc': acc_sensible,
              'H_abl': abl_sensible,
              'B_acc': acc_ground,
              'B_abl': abl_ground,
              'TS_acc': acc_ts,
              'TS_abl': abl_ts}

    results_list.append(result)

results_df = pd.DataFrame(results_list)
results_df.to_csv("/data/scratch/richteny/thesis/cosipy_test_space/LHS_1D20m_1999_2010_EBfromLHS.csv")
