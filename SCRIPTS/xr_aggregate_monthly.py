import xarray as xr
import numpy as np
import pathlib
import datetime
import pandas as pd

datapath="/data/scratch/richteny/thesis/io/data/output/bestfiles/"
outpath="/data/scratch/richteny/thesis/io/data/output/current/bestfiles/"

def spatial_mean(ds):
    weights = ds['N_Points'] / np.sum(ds['N_Points'])
    result = ds.copy()
    for var in list(ds.variables):
        if var not in ["time", "lat", "lon", "HGT", "MASK", "SLOPE", "ASPECT", "N_Points"]:
            weighted_field = (ds[var] * weights).sum(dim=['lat', 'lon'])
            result[var] = weighted_field
    return result

start = datetime.datetime.now()

#slice has duplicates, check why that is? Were does it come from? 
#full_range = pd.date_range("2009-01-01", "2020-01-01", freq="1H")
#Load sample ds
#sample = xr.open_dataset(datapath+"HEF_COSMO_1D20m_1999_2010_HORAYZON_LHS_19990101-20091231_RRR-1.2982_0.7606_0.3084_0.6344_2.1036_9.2323_0.24_1.7_4.0_0.0026_num2.nc") 
sample = xr.open_dataset(datapath+"HEF_COSMO_1D20m_1999_2010_HORAYZON_IntpPRES_MCMC-ensemble_19990101-20091231_RRR-0.7409_0.8852_0.2279_0.6388_13.82_1.0188_0.24_1.7216_4.0_0.0026_num274.nc")
sample = sample.sel(time=slice("2000-01-01","2009-12-31"))
mb_holder = pd.DataFrame(index=sample.time)

for fp in pathlib.Path(datapath).glob('HEF_COSMO_1D20m_1999_2010_HORAYZON_IntpPRES*.nc'):
    print(fp)
    ds = xr.open_dataset(fp)
    #for cumulative mass balances create timeseries of MB as dataframe?
    #spat_mean = ds[['MB']].sel(time=slice("2000-01-01","2009-12-31")).mean(dim=['lat','lon'])
    ds = ds[['G','ALBEDO','LWin','LWout','ME', 'TS', 'H','LE','B','QRR','N_Points','MB']].sel(time=slice("2000-01-01","2009-12-31"))
    spat_mean = spatial_mean(ds)
    
    raw_fp = str(fp.stem).split('HEF_COSMO_1D20m_1999_2010_HORAYZON_IntpPRES_MCMC-ensemble_19990101-20091231_RRR-')[-1]
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
    name = f"{rrr_factor}_{alb_snow}_{alb_ice}_{alb_firn}_{alb_aging}_{alb_depth}_{roughness_ice}"

    mb_holder[name] = spat_mean.MB.values
    #create monthly data 
    monthly_ds = spat_mean.groupby('time.month').mean()
    monthly_ds.to_netcdf(outpath+'AvgMonthly_'+str(fp.name))

mb_holder.to_csv(outpath+"AvgSpatMB_ens.csv")
print("Done after", datetime.datetime.now()-start)
