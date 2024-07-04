import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import salem
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

## Load paths ##
rf_path = "/data/scratch/richteny/thesis/AWS/Abramov/RF/"
outpath = "/data/scratch/richteny/thesis/cosipy_test_space/utilities/createforcing/"
glacier_outline = "../../data/static/Shapefiles/abramov_rgi6.shp"
dem = "/data/projects/topoclif/input-data/DEMs/HMA_alos-jaxa.tif"


#Helper Functions #
#Heavily influenced by https://gis.stackexchange.com/questions/260304/extract-raster-va$
def get_glacier_elev(dem):
    gla_shp = salem.read_shapefile(glacier_outline)
    geoms = gla_shp.geometry.values
    geometry = geoms[0]
    geoms = [mapping(geoms[0])]
    #this assumes there is one continuous polygon
    with rasterio.open(dem) as src:
        out_image, out_transform = mask(src, geoms, crop=True)
    no_data=src.nodata
    data = out_image[0,:,:]
    elev = np.extract(data != no_data, data)
    elev = elev[elev != no_data]
    print(elev)
    mean_alt = np.nanmean(elev)
    return mean_alt

def datetime_index(df, time_var, rename, old_time_name=None):
    if rename:
        df[time_var] = pd.to_datetime(df.pop(old_time_name))
    else:
        df[time_var] = pd.to_datetime(df[time_var])
    df.set_index(time_var, inplace=True)
    return df

spinup = True
## Load data ##
#Load raw ERA5 forcing
era5_df = pd.read_csv(outpath+"Abramov_ERA5_1999_2021.csv", index_col=0)
era5_df = datetime_index(era5_df, 'time', True, 'TIMESTAMP')
era5_df = era5_df.loc["2010-01-01":"2020-01-01"]
print(era5_df)

#Load lapse rates or just give values at glacier elevation?

mean_glacier_alt = get_glacier_elev(dem)
print(mean_glacier_alt)
aws_hgt = 4102 #Kronenberg at al. 2022 says 4100 

for col in ["RH2","T2","U2","PRES"]:
    print(col)
    # Load the prediction # 
    pred = pd.read_hdf(rf_path+'{}_predict-full_timeseries.h5'.format(col))
    pred.reset_index(inplace=True)
    pred = datetime_index(pred, 'time', True, 'date')
    #Need to scale data to mean glacier elevation -> RF regr. at level of AWS or just give elevation at AWS level. 
    print(pred)
    era5_df[col] = pred['{}_pred_ensemble'.format(col)]

print(era5_df)

print("------------------------------\n")
#Extend timeseries with copy of first year to create 1-year spinup
if spinup:
    cp_year = era5_df.loc["2010-01-01":"2011-01-01"]
    print(cp_year)
    cp_year.reset_index(inplace=True)
    print(cp_year)
    cp_year['time'] = pd.to_datetime(cp_year['time'] - pd.DateOffset(years=1))
    cp_year.set_index(['time'], inplace=True)
    full_df = pd.concat([cp_year, era5_df])
    print(full_df.isnull().sum())
    full_df.index.names = ['TIMESTAMP']
    full_df.reset_index(inplace=True) #I think the rest of the processing needs it
    full_df.to_csv("../../data/input/Abramov/Abramov_ERA5mod_spinup_Forcing_2009-2020.csv")
else:
    era5_df.index.names = ['TIMESTAMP']
    era5_df.reset_index(inplace=True)
    era5_df.to_csv("../../data/input/Abramov/Abramov_ERA5mod_nospinup_Forcing_2010-2020.csv")

    
