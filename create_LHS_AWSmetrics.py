import pandas as pd
import numpy as np
import pathlib
import xarray as xr

path = "/data/scratch/richteny/thesis/cosipy_test_space/data/output/LHS/"
awspath = "/data/scratch/richteny/thesis/Hintereisferner/Climate/AWS_Obleitner/"

aws = pd.read_csv(awspath+"Fix_HEFupper_01102003_24102004.csv", index_col="time", parse_dates=True)
print(aws.isnull().sum())

def get_closest_elevgridcell(ds, elevation_target=3048.0):
    
    #Recorded original coordinates could not be converted into lat/lon coordinates due to unknown projection.
    #Reconstructed positions from later recorded GPS points of AWS locations (+/-100 m):
    #HEF lower: 46.813570° N; 10.788977° E; 2640 m
    #HEF upper: 46.790453° N; 10.747121° E; 3048 m
    abs_diff = abs(ds['HGT'] - elevation_target)
    closest_cells = abs_diff.where(abs_diff == abs_diff.min(), drop=True)
    lat = closest_cells.lat
    lon = closest_cells.lon
    return ds.sel(lat=lat, lon=lon)

def get_metrics(obs, mod):
    abs_error = abs(mod - obs)
    sq_error = (mod - obs)**2
    mse = sq_error.mean()
    rmse = np.sqrt(mse)
    mae = abs_error.mean()
    
    return (mae, rmse)

daily_sum = aws[['SWO','SWI']].resample('D').sum()
daily_albedo = daily_sum['SWO'] / daily_sum['SWI']
daily_albedo = daily_albedo.where((daily_albedo >= 0.) & (daily_albedo <= 1.))
daily_albedo.rename('albedo', inplace=True)

daily_mean = aws.resample("D").mean()
daily_sfc = daily_mean['sfc']
first_value = daily_sfc[0].copy() #store for adjustment of COSIPY TOTALHEIGHT
print(first_value)
daily_sfc = daily_sfc - daily_sfc[0]

daily_lwo = daily_mean['LWO']

joint_df = pd.concat([daily_lwo, daily_albedo, daily_sfc], axis=1)
## Loop through files

results_list = []

for fp in pathlib.Path(path).glob('*.nc'):
    print(fp.stem)
    ds = xr.open_dataset(fp).sel(time=slice("2003-10-01","2004-10-24"))[['ALBEDO','LWout','TOTALHEIGHT','HGT']]
    sub = get_closest_elevgridcell(ds, elevation_target=3048.0).resample(time="1D").mean().isel(lat=0,lon=0)
    sub['LWout'] = sub['LWout'] * -1
    offset = first_value - sub['TOTALHEIGHT'][0]
    totalheight = sub['TOTALHEIGHT'] + offset
    norm_totalheight = totalheight - totalheight[0]
    
    #get metrics
    mae_albedo, rmse_albedo = get_metrics(daily_albedo, sub.ALBEDO.values)
    print(mae_albedo, rmse_albedo)
    
    mae_lwout, rmse_lwout = get_metrics(daily_lwo, sub.LWout.values)
    print(mae_lwout, rmse_lwout)
    
    mae_sfc, rmse_sfc = get_metrics(daily_sfc, norm_totalheight.data)
    print(mae_sfc, rmse_sfc)

    #Create df with timeseries and store
    holder = joint_df.copy()
    holder['cspy_alb'] = sub.ALBEDO.values
    holder['cspy_sfc'] = norm_totalheight.data
    holder['cspy_lwout'] = sub.LWout.values
    filename = str(fp.stem) + "AWSmetrics.csv"
    holder.to_csv("/data/scratch/richteny/thesis/cosipy_test_space/data/output/LHS/AWScomp/"+filename)
    del holder 

    #get params from filename
    raw_fp = str(fp.stem).split('HEF_COSMO_1D20m_1999_2010_HORAYZON_IntpPRES_LHS-wide_19990101-20091231_RRR-')[-1]
    rrr_factor = float(raw_fp.split('_')[0])
    alb_snow = float(raw_fp.split('_')[1])
    alb_ice = float(raw_fp.split('_')[2])
    alb_firn = float(raw_fp.split('_')[3])
    alb_aging = float(raw_fp.split('_')[4])
    alb_depth = float(raw_fp.split('_')[5])
    roughness_ice = float(raw_fp.split('_')[7])
    
    result = {'rrr_factor': rrr_factor,
              'alb_ice': alb_ice,
              'alb_snow': alb_snow,
              'alb_firn': alb_firn,
              'albedo_aging': alb_aging,
              'albedo_depth': alb_depth,
              'roughness_ice': roughness_ice,
              'mae_alb': mae_albedo,
              'rmse_alb': rmse_albedo,
              'mae_lwout': mae_lwout,
              'rmse_lwout': rmse_lwout,
              'mae_sfc': mae_sfc,
              'rmse_sfc': rmse_sfc}
    
    results_list.append(result)

results_df = pd.DataFrame(results_list)
results_df.to_csv("/data/scratch/richteny/thesis/cosipy_test_space/LHS_1D20m_1999_2010_AWSmetrics.csv")
