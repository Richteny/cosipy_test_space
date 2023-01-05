import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
import pathlib
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.metrics import r2_score
import salem

## ----- Load paths and data ----- ##
#era_path = "/data/scratch/richteny/ERA5_HMA/"
#plot_path = "/data/scratch/richteny/thesis/cosipy_test_space/utilities/createforcing/"
#glacier_outline = "../../data/static/Shapefiles/parlung4_rgi6.shp"

#g = 9.80665 #m/s**2 gravitational acceleration
#vars to consider
#vars_to_consider = ['t2m','d2m','tp']

#1. Get lat lon of nearest ERA5 cell where glacier is in, warning does not account for bordering cells etc.
def select_nearest_latlon(era_path, glacier_outline):
    era5_gp = salem.open_xr_dataset(era_path+"ERA5_HMA_15N50N_60E125E_1999_2021_z.nc")
    #Assumes files are in same projection and all ERA5 files have same grid
    shape_grid = salem.read_shapefile_to_grid(glacier_outline, grid=salem.grid_from_dataset(era5_gp))
    lon_distance = np.abs(era5_gp.longitude.values-shape_grid.CenLon.values)
    lat_distance = np.abs(era5_gp.latitude.values-shape_grid.CenLat.values)
    idx_lon = np.where(lon_distance == np.nanmin(lon_distance))
    idx_lat = np.where(lat_distance == np.nanmin(lat_distance))
    latitude = float(era5_gp.latitude[idx_lat].values)
    longitude = float(era5_gp.longitude[idx_lon].values)
    return latitude, longitude

def calculate_lapse_rates(glacier_outline):

    era_path = "/data/scratch/richteny/ERA5_HMA/"
    plot_path = "/data/scratch/richteny/thesis/cosipy_test_space/utilities/createforcing/"
    print("Getting lapse rates for", glacier_outline)
    g = 9.90665 #m/s**2 gravitational acceleration
    vars_to_consider = ['t2m','d2m']
    
    lat, lon = select_nearest_latlon(era_path, glacier_outline)
    print(lat,lon)

    #Load static data#
    ds_static = salem.open_xr_dataset(era_path+"ERA5_HMA_15N50N_60E125E_1999_2021_z.nc")
    ds_static = ds_static.sel(latitude=slice(lat+1.25,lat-1.25), longitude=slice(lon-1.25,lon+1.25))
    glacier = salem.read_shapefile(glacier_outline)

    #Create 1 degree buffer centered around nearest latlon from glacier
    #Create plot
    fig, ax = plt.subplots(1,1, figsize=(16,9))
    ds_static['HGT'] = ds_static['z'][0,:,:] / g
    glacier.plot(ax=ax)
    #print(ds_static['HGT'])
    ds_static['HGT'].plot(ax=ax, zorder=-1)
    plt.savefig(plot_path+"glacier_in_static_test.png")

    #Label Dic
    label_dict= {'d2m': 'Dewpoint temperature at 2m (°C)',
                 't2m': 'Air temperature at 2m (°C)',
                 'tp-diff': 'Total precipitation (mm)'}

    title_dict= {'d2m': 'Dewpoint temperature',
                 't2m': 'Air temperature',
                 'tp-diff': 'Total precipitation'}
    dic_slopes = {}

    for var in vars_to_consider:
        fp = pathlib.Path(era_path).glob('*1999_2021_{}.nc'.format(var))
        filepath = list(fp)
        print(filepath)
        ds = salem.open_xr_dataset(str(filepath[0]))
        ds = ds.sel(latitude=slice(lat+1.25,lat-1.25), longitude=slice(lon-1.25,lon+1.25))
        #radiation and precipitation are cumulative
        if var == 'tp':
            ds['tp-diff'] = ds[var].diff(dim='time')
            ds['tp-diff'][0] = ds[var][0]
            ds['tp-diff'] = ds['tp-diff'].where(ds['tp-diff'] >= 0, ds[var]) #cond = location where to preserve values
            ds['tp-diff'] = ds['tp-diff'].where(ds['tp-diff'] >= 0, 0)
            ds = ds[['tp-diff']]
        #print(ds)
        mean_ds = ds.mean(dim='time')
        mean_df = mean_ds.to_dataframe()
        mean_df.reset_index(inplace=True)
        print(mean_df)
        static_df = ds_static[['HGT']].to_dataframe()
        static_df.reset_index(inplace=True)
        print(static_df) 
        #static_df.drop('time', axis=1, inplace=True)
        print(static_df)
        #Merge data
        merged_df = pd.merge(mean_df, static_df, how='left', left_on= ['latitude','longitude'], right_on = ['latitude','longitude'])
        print(merged_df)
        
        #Start plot
        for var in merged_df.columns:
            if var not in ['z','longitude','latitude','HGT']:
                if var in ['t2m','d2m']:
                    merged_df[var] = merged_df[var]-273.15
            
                plt.rc('axes', axisbelow=True)
                b, m = polyfit(merged_df['HGT'], merged_df[var], 1)
                pred = b + m*merged_df['HGT']
                r2score = r2_score(merged_df[var],pred)
                #Scatter Plot
                fig = plt.figure(figsize=(10,10), dpi=300)
                ax = fig.add_subplot(111)
                #params = {'legend.fontsize': 18,
                #          'legend.handlelength': 1}
                #plt.rcParams.update(params)
                ax.scatter(merged_df['HGT'], merged_df[var], color='k')
                ax.plot(merged_df['HGT'], b + m * merged_df['HGT'], '-')
                ax.tick_params(axis='both', which='major', labelsize=18)
                ax.set_xlabel('ERA5 elevation (m a.s.l.)',fontsize=20)
                ax.set_ylabel(label_dict[var],fontsize=20)
                plt.autoscale(enable=True, axis='y')
                ax.grid()
                ax.set_title('Slope: {}, $R^2$: {}'.format(round(m, 6), round(r2score,4)), fontsize=20)
                if var in ['tp-diff']:
                    col = var.split('-')[0]
                    print(col)
                    fig.savefig(plot_path+col+'-lapse-rate-300dpi.png', bbox_inches='tight')
                else:
                    fig.savefig(plot_path+var+'-lapse-rate-300dpi.png', bbox_inches='tight')
                plt.close(fig)
                dic_slopes[var] = m
        print(dic_slopes)

    return (dic_slopes['t2m'],dic_slopes['d2m'])
           

if '__name__' == '__main__':
    glacier_outline = "../../data/static/Shapefiles/parlung4_rgi6.shp"
    calculate_lapse_rates(glacier_outline)
