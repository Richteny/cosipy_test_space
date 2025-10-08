#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 19:18:42 2023

@author: niki
"""

import xarray as xr
import numpy as np
import pandas as pd
import pathlib 
import matplotlib.pyplot as plt

#setting
value_choice = 'elevation' #nearest

path = "/data/scratch/richteny/thesis/io/data/output/bestfiles/"
outpath = "/data/scratch/richteny/thesis/io/data/output/aws_comp/bestfiles/"

aws_path = "/data/scratch/richteny/thesis/Hintereisferner/Climate/AWS_Obleitner/"
aws_lower = pd.read_csv(aws_path+"Fix_HEFlower_01102003_24102004.csv", parse_dates=True, index_col="time")
aws_upper = pd.read_csv(aws_path+"Fix_HEFupper_01102003_24102004.csv", parse_dates=True, index_col="time")

def get_closest_elevgridcell(ds, elevation_target=3048.0):
    
    #Recorded original coordinates could not be converted into lat/lon coordinates due to un$
    #Reconstructed positions from later recorded GPS points of AWS locations (+/-100 m):
    #HEF lower: 46.813570° N; 10.788977° E; 2640 m
    #HEF upper: 46.790453° N; 10.747121° E; 3048 m
    abs_diff = abs(ds['HGT'] - elevation_target)
    closest_cells = abs_diff.where(abs_diff == abs_diff.min(), drop=True)
    lat = closest_cells.lat
    lon = closest_cells.lon
    return ds.sel(lat=lat, lon=lon)

#Load simulations                        
vars_of_interest = ['TOTALHEIGHT','G','SNOWHEIGHT','MB','surfMB','HGT','MASK','RAIN', 'SNOWFALL', 'RRR', 'ME','surfM','subM', 'LWout','LWin', 'TS', 'ALBEDO', 'T2', 'RH2', 'U2', 'PRES' ]
#vars_of_interest = ["TOTALHEIGHT","T2","RH2","LWin","G"]
def process_netcdf_files(file, location):
        
    if location == "lower":
        ds = file[vars_of_interest]
        sub = get_closest_elevgridcell(ds, elevation_target=2640.0).isel(lat=0,lon=0)
        sub = sub.where(sub.time.isin(aws_lower.index), drop=True)
        print(sub)
        offset = aws_lower['sfc'][0] - sub['TOTALHEIGHT'][0]
        fixed_totalheight = sub['TOTALHEIGHT'] + offset
        norm_totalheight = fixed_totalheight - fixed_totalheight[0]
        sub['TOTALHEIGHT'] = norm_totalheight
        #instead of unstack we could use the mean
        #compute SWnet here
        sub['SWnet'] = sub['G'] * (1 - sub['ALBEDO'])
        df = sub.to_dataframe()
        
    elif location == "upper":
        ds = file[vars_of_interest]
        sub = get_closest_elevgridcell(ds, elevation_target=3048.0).isel(lat=0,lon=0)
        sub = sub.where(sub.time.isin(aws_upper.index), drop=True)
        offset = aws_upper['sfc'][0] - sub['TOTALHEIGHT'][0]
        fixed_totalheight = sub['TOTALHEIGHT'] + offset
        norm_totalheight = fixed_totalheight - fixed_totalheight[0]
        sub['TOTALHEIGHT'] = norm_totalheight
        sub['SWnet'] = sub['G'] * (1 - sub['ALBEDO'])
        df = sub.to_dataframe()
    else:
        print("Failure, did not recognize parameters.")
            
    return df
            

dic_aws_u = {}
dic_aws_l = {}
for fp in pathlib.Path(path).glob('HEF_COSMO_*.nc'):
    print(fp)
    #key = str(fp.stem).split('_')[-1].split('.nc')[0] #get number
    #key = str(fp.stem).split('_')[5] + str(fp.stem).split('_')[6]
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
    key = f"{rrr_factor}_{alb_snow}_{alb_ice}_{alb_firn}_{alb_aging}_{alb_depth}_{roughness_ice}"
    file = xr.open_dataset(fp) #melt, whatever variables we need, LWout albedo etc. 
    #select same elevation band or closest grid cell
    dic_aws_l[key] = process_netcdf_files(file, location="lower")
    dic_aws_u[key] = process_netcdf_files(file, location="upper")


#now process files
keys = list(dic_aws_l.keys())

# Loop over each relevant variable and merge them into one dataframe
for var in (vars_of_interest + ['SWnet']):
    if var not in ["HGT","MASK"]:
        print(var)
        df_upper = dic_aws_u[keys[0]][[var]].copy()
        df_upper.rename(columns={var:keys[0]}, inplace=True)
        #
        df_lower = dic_aws_l[keys[0]][[var]].copy()
        df_lower.rename(columns={var:keys[0]}, inplace=True)

        for key in keys[1:]:
            df_upper[key] = dic_aws_u[key][var]
            df_lower[key] = dic_aws_l[key][var]
    
    df_upper.to_csv(outpath+"cosipy_at_upperaws_grid_merged-{}.csv".format(var))
    df_lower.to_csv(outpath+"cosipy_at_loweraws_grid_merged-{}.csv".format(var))
    

#NEXT develop plotting script!
#Then, what else is missing?
'''
Recorded original coordinates could not be converted into lat/lon coordinates due to unknown projection.
Reconstructed positions from later recorded GPS points of AWS locations (+/-100 m):
HEF lower: 46.813570° N; 10.788977° E; 2640 m
HEF upper: 46.790453° N; 10.747121° E; 3048 m
'''
