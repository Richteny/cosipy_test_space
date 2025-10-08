import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
plt.rcParams.update({'font.size': 22})

def prepare_ds(ds):
    ds = ds[['MB','SNOWFALL','RRR','N_Points','REFREEZE','T2', 'HGT','MASK']].sel(time=slice("2000-01-01","2009-12-31"))
    #density_fresh_snow = np.maximum(109.0 + 6.0 * (ds['T2'].values - 273.16) + 26.0 * np.sqrt(ds['U2'].values), 50.0)
    #ds['density_fresh_snow'] = (('time','lat','lon'), density_fresh_snow)
    #ice_density = 917.
    #ds['SF'] = ds['SNOWFALL']*(ds['density_fresh_snow']/ice_density)*1000
    #set everywhere outside of glacier to nan
    #ds = ds.where(ds.mask==1)
    print("Finished preparing dataset.")
    return ds

path = "/data/scratch/richteny/thesis/io/data/output/bestfiles/"
outpath= "/data/scratch/richteny/thesis/io/data/output/hypso/bestfiles/"

def get_area_weighted_elevation_stats(ds, time_slices_wy, year_vals):
    """
    Calculates area-weighted statistics for specified variables and elevation bands.

    This function correctly resamples the data from its original resolution (e.g., 20m bands)
    to a new resolution (e.g., 50m bands) by weighting each original band's contribution
    by its area, represented here by 'N_Points'.
    """
    # --- Configuration ---
    # Define the new 50m elevation bins and their labels
    bins = np.arange(2400, 3700 + 50, 50)
    labels = bins[:-1] + 25
    
    # Variables to process and how to aggregate them annually
    # 'sum' for cumulative values like mass balance, 'mean' for state values like temperature.
    variables_to_process = {
        'MB': 'sum',
        'SNOWFALL': 'sum',
        'T2': 'mean'
    }
    
    # --- Data Preparation ---
    # Create a base DataFrame with elevation and weights (N_Points).
    # We assume the 'lat' dimension corresponds to the elevation bands.
    # The .squeeze() method removes any singleton dimensions (like 'lon').
    base_df = pd.DataFrame({
        'HGT': ds['HGT'].squeeze().values,
        'N_Points': ds['N_Points'].squeeze().values
    })
    
    # Assign each original elevation band to one of the new 50m bins.
    base_df['HGT_bins'] = pd.cut(base_df['HGT'], bins=bins, labels=labels, include_lowest=True, right=False)

    # --- Processing Loop ---
    # This will hold the final results for all years.
    all_years_stats = []

    for i, tslice in enumerate(time_slices_wy):
        print(f"Processing year: {year_vals[i]} ({tslice[0]} to {tslice[1]})")
        
        # Create a temporary DataFrame for the current year's data
        temp_df = base_df.copy()
        
        # Calculate the annual value for each variable
        for var, method in variables_to_process.items():
            annual_data = ds[var].sel(time=slice(tslice[0], tslice[1]))
            
            if method == 'sum':
                # Sum over the time dimension for the water year
                temp_df[var] = annual_data.sum(dim='time', skipna=True, min_count=1).squeeze().values
            elif method == 'mean':
                # Mean over the time dimension for the water year
                temp_df[var] = annual_data.mean(dim='time', skipna=True).squeeze().values

        # --- Area-Weighted Aggregation ---
        # Group by the new 50m bins and calculate the weighted average for each variable.
        
        # Define the weighted average function
        def weighted_avg(group):
            results = {}
            weights = group['N_Points']
            total_weight = weights.sum()
            if total_weight == 0:
                # Return NaN if there's no area/points in the bin
                for var in variables_to_process:
                    results[var] = np.nan
                return pd.Series(results)

            for var in variables_to_process:
                weighted_sum = (group[var] * weights).sum()
                results[var] = weighted_sum / total_weight
            return pd.Series(results)

        # Apply the function
        yearly_stats = temp_df.groupby('HGT_bins').apply(weighted_avg).reset_index()
        yearly_stats['year'] = year_vals[i]
        all_years_stats.append(yearly_stats)

    # Combine the results from all years into a single DataFrame
    final_stats = pd.concat(all_years_stats, ignore_index=True)
    
    # Pivot to have elevation bands as the index and variables as columns
    # Here we average across all processed years. If you want separate years, adjust this step.
    final_pivot = final_stats.groupby('HGT_bins')[list(variables_to_process.keys())].mean()
    
    return final_pivot
"""
def get_elevation_stats(ds):
    hgt = ds.HGT.values
    mask = ds.MASK.values
    bins = np.arange(2400,3700+50,50) #from WGMS bins
    labels= bins[:-1]+25
    #year_vals = np.arange(2002,2009+1,1) #2001
    ##time_slices_wy= [('2000-10-01','2001-09-30'),('2001-10-01','2002-09-30'),('2002-10-01','2003-09-30'),('2003-10-01','2004-09-30'),
    #time_slices_wy = [('2001-10-01','2002-09-30'),('2002-10-01','2003-09-30'),('2003-10-01','2004-09-30'),
    #                 ('2004-10-01','2005-09-30'),('2005-10-01','2006-09-30'),('2006-10-01','2007-09-30'),('2007-10-01','2008-09-30'),('2008-10-01','2009-09-30')]

    year_vals = np.array([2004])
    time_slices_wy = [("2003-10-01","2004-09-30")]
    
    elev_stats = pd.DataFrame(index=labels)
    var_of_interest = ['MB','SNOWFALL','T2']
    ## Summarise per Elevation Band ##

    for var in var_of_interest:
    #mean annual?
        test = ds[var].where(ds.MASK==1).groupby_bins(ds.HGT, bins, labels=labels, include_lowest=True).mean(skipna=True)
        mb_holder = np.zeros(shape=(len(year_vals),test.HGT_bins.shape[0]))
        print(mb_holder.shape)
        i=0
        if var == 'MB':
            print(var)
            for tslice in time_slices_wy:
                print(tslice)
                mb_holder[i,:] = test.sel(time=slice(tslice[0],tslice[1])).sum(dim='time', skipna=True, min_count=1)
                i+=1
        elif var in ['RRR','SNOWFALL','SF','REFREEZE']:
            for tslice in time_slices_wy:
                print(tslice)
                mb_holder[i, :] = test.sel(time=slice(tslice[0], tslice[1])).sum(dim='time', skipna=True, min_count=1)
                i += 1
        else:
            for tslice in time_slices_wy:
                print(tslice)
                mb_holder[i, :] = test.sel(time=slice(tslice[0], tslice[1])).mean(dim='time', skipna=True)
                i += 1
        mean_per_elevband = np.nanmean(mb_holder, axis=0)
        elev_stats[var] = mean_per_elevband
    return elev_stats
"""
##for debug
##list_vals = ['/data/scratch/richteny/thesis/cosipy_test_space/data/output/sim/TEST_MPI_19990101-20100101_num93_lrT_0.0_lrRRR_5.16e-05_prcp_0.825_albsnow_0.8506.nc',
##             '/data/scratch/richteny/thesis/cosipy_test_space/data/output/sim/TEST_MPI_19990101-20100101_num9_lrT_0.0_lrRRR_0.0001891_prcp_0.6836_albsnow_0.888.nc']
#year_vals = np.arange(2002,2009+1,1) #2001
##time_slices_wy= [('2000-10-01','2001-09-30'),('2001-10-01','2002-09-30'),('2002-10-01','2003-09-30'),('2003-10-01','2004-09-30'),
#time_slices_wy = [('2001-10-01','2002-09-30'),('2002-10-01','2003-09-30'),('2003-10-01','2004-09-30'),
#                 ('2004-10-01','2005-09-30'),('2005-10-01','2006-09-30'),('2006-10-01','2007-09-30'),('2007-10-01','2008-09-30'),('2008-10-01','2009-09-30')]

year_vals = np.array([2004])
time_slices_wy = [("2003-10-01","2004-09-30")]

for fp in pathlib.Path(path).glob('HEF_COSMO_1D20m_1999_2010_HORAYZON_IntpPRES*.nc'):
    ds = xr.open_dataset(fp)
    ds = prepare_ds(ds)
    print(fp)
    #key = str(fp.stem).split('_')[-1].split('.nc')[0]
    filename = str(fp.stem)
    #elev_stats = get_elevation_stats(ds)
    elev_stats = get_area_weighted_elevation_stats(ds, time_slices_wy, year_vals)
    print(elev_stats)
    #produces this per run
    elev_stats.to_csv(outpath+"hypsometry_cspy_YEAR2004_{}.csv".format(filename))
    #elev_stats.to_csv(outpath+"hypsometry_cspy_{}.csv".format(filename))

##get area per elevation bin
#area_ds = ds.MASK.groupby_bins(ds.HGT, bins, labels=labels, include_lowest=True).sum(skipna=True,min_count=1)
#area_per_bin = area_ds.values * 0.3 * 0.3

'''
#Last possible run test another lapse rate
print("Finished calculating elevation stats.")

with plt.xkcd():
    fig, axes = plt.subplots(1,3, figsize=(16,9), dpi=300, sharey=True)
    axes[0].plot(elev_stats.MB, elev_stats.index, marker='o', color="black", linewidth=1.5, zorder=6)
    axes[0].plot(elev_stats2.MB, elev_stats2.index, marker='o', color="blue", linewidth=1.5, zorder=6)
    axes[0].plot(elev_stats3.MB, elev_stats3.index, marker='o', color="green", linewidth=1.5, zorder=6)
    axes[0].plot(elev_stats4.MB, elev_stats4.index, marker='o', color="orange", linewidth=1.5, zorder=6)
    axes[0].axvline(x=0, linestyle='--', color="black")
    axes[0].axhline(y=4100, linestyle='--', color="red")
    axes[0].set_xlim(-6,2)
    axes[0].set_ylim(3650,4850)
    axes[0].set_xticks(np.arange(-5,2+0.5,0.5))
    axes[0].set_yticks(np.arange(3650,4850+50,100))
    axes[0].set_xticklabels(axes[0].get_xticks(), rotation=45)
    axes[0].set_xlabel("Mass Balance (m w.e. a$^{-1}$)")
    ax2 = axes[0].twiny()
    ax2.barh(elev_stats.index, area_per_bin,align='center', height=30, color="blue", alpha=0.3, edgecolor="white", zorder=-1)
    ax2.set_xticks(np.arange(0,3.5+0.5,0.5))
    ax2.set_xlim(0,3.5)
    ax2.set_xlabel("Area (km²)")
    axes[0].set_ylabel("Elevation (m a.s.l.)")
    axes[0].grid()
    #Temp.
    axes[1].plot(elev_stats.T2, elev_stats.index, marker='o', color="black", linewidth=1.5)
    axes[1].plot(elev_stats4.T2, elev_stats4.index, marker='o', color="orange", linewidth=1.5)
    axes[1].axhline(y=4100, linestyle='--', color="red")
    axes[1].axvline(x=273.15, linestyle='--', color="black")
    #axes[1].set_xlim(-2, 2.25)
    #axes[1].set_ylim(3650, 4850)
    #axes[1].set_xticks(np.arange(-2, 2 + 0.25, 0.25))
    #axes[1].set_yticks(np.arange(3650, 4850 + 50, 100))
    axes[1].set_xlabel("MAAT at 2m (K)")
    #SF + RRR, drop SF for now - conversion seems wrong somehow
    axes[2].plot(elev_stats.RRR, elev_stats.index, marker='o', color="black", linewidth=1.5)
    #axes[2].plot(elev_stats.SF, elev_stats.index, marker='.', linestyle='--', color="black", alpha=0.5, linewidth=1.5)
    axes[2].plot(elev_stats2.RRR, elev_stats2.index, marker='o', color="blue", linewidth=1.5)
    #axes[2].plot(elev_stats2.SF, elev_stats2.index, marker='.', linestyle='--', color="blue", alpha=0.5, linewidth=1.5)
    axes[2].plot(elev_stats3.RRR, elev_stats3.index, marker='o', color="green", linewidth=1.5)
    #axes[2].plot(elev_stats3.SF, elev_stats3.index, marker='.', linestyle='--', color="green", alpha=0.5, linewidth=1.5)
    axes[2].plot(elev_stats4.RRR, elev_stats4.index, marker='o', color="orange", linewidth=1.5)
    #axes[2].plot(elev_stats4.SF, elev_stats4.index, marker='.', linestyle='--', color="orange", alpha=0.5, linewidth=1.5)
    axes[2].axhline(y=4100, linestyle='--', color="red")
    #axes[1].set_xlim(-2, 2.25)
    axes[2].set_ylim(3650, 4850)
    axes[2].set_xticks(np.arange(0, 1750 + 250, 250))
    axes[2].set_yticks(np.arange(3650, 4850 + 50, 100))
    axes[2].set_xlabel("Precipitation (mm a$^{-1}$)")
    plt.savefig("E:/OneDrive - uibk.ac.at/PhD/PhD/Data/Slides_Preparation/output/" + 'glacier_elevation_plots4.png',
                bbox_inches="tight")
    plt.show()

### What else do we need for the story line? ###
#A plot of the transient snowline altitudes throughout the year and the glacier TSLA
#A plot of the temperature distribution and precipitation distribution
# ?


#Plot MB over time, how to implement the geod. MB
#
geod = pd.read_csv("E:/OneDrive - uibk.ac.at/PhD/PhD/Data/Hugonnet_21_MB/time_series_13/dh_abramov_pergla_rates.csv")
geod = geod.loc[geod['period']=='2010-01-01_2020-01-01']
mb_geod = np.array([geod['dmdtda']]*11).ravel()
err_geod = np.array([geod['err_dmdtda']]*11).ravel()
#
#list geod. MB
def prepare_geod_mb(ds):
    spat_mean = ds.MB.mean(dim=['lat','lon'], skipna=True)
    mb_holder = np.zeros(shape=(len(year_vals)))
    i=0
    for tslice in time_slices_fy:
        print(tslice)
        mb_holder[i] = spat_mean.sel(time=slice(tslice[0], tslice[1])).sum(dim='time', skipna=True, min_count=1)
        i+=1
    mb_mod = np.array([np.nanmean(mb_holder)] * 11)
    print(np.nanmean(mb_holder))
    return mb_mod
mb_mod = prepare_geod_mb(ds)
mb_mod2 = prepare_geod_mb(ds2)
mb_mod3 = prepare_geod_mb(ds3)
mb_mod4 = prepare_geod_mb(ds4)

#in this figure we could add transient snowline altitudes over time and comparison just to see fit
#Un-normalize TSLA values
#add to plot
tsla_obs = pd.read_csv(path+"TSLA_Abramov_filtered_full.csv", index_col="LS_DATE", parse_dates=True)
tsla_obs = tsla_obs.loc["2010-01-01":"2019-12-31"]

def process_tsla_values(filename):
    tsla_mod = pd.read_csv(path+filename, index_col="time", parse_dates=True)
    #unnormalize
    min_elev = np.min(hgt[mask == 1])
    max_elev = np.max(hgt[mask == 1])
    tsla_mod['SC_med'] = tsla_mod['Med_TSL'] * (max_elev - min_elev) + min_elev
    tsla_mod['SC_std'] = tsla_mod['Std_TSL'] * (max_elev - min_elev) + min_elev
    tsla_mod = tsla_mod[tsla_mod.index.isin(tsla_obs.index)]
    return tsla_mod
tsla_mod = process_tsla_values("tsla_abramov_300m_era5mod_wohlfahrt_20090101-20200101_num_lrt_0.0_lrrrr_0.0_prcp_1.25.csv")
tsla_mod2 = process_tsla_values("tsla_abramov_300m_era5mod_wohlfahrt_20090101-20200101_num_lrt_0.0_lrrrr_7e-05_prcp_1.25_albsnow_0.85.csv")
tsla_mod3 = process_tsla_values("tsla_abramov_300m_era5mod_wohlfahrt_20090101-20200101_num_lrt_0.0_lrrrr_0.00018_prcp_1.25.csv")
tsla_mod4 = process_tsla_values("tsla_abramov_300m_era5mod_wohlfahrt_20090101-20200101_num_lrt_0.007148_lrrrr_0.00018_prcp_1.25.csv")

#alb 0.65 -> -2 mb with 1.5 factor
#alb 0.75 - -0.7 mb with 1.5 factor
#now alb 0.8 with 1.5 factor -> -0.146
#can also go other way and increase prec. to get to MB

with plt.xkcd():

    fig, axes = plt.subplots(2,1, figsize=(16,9), dpi=300)
    axes[0].plot(np.arange(2010,2020+1,1), mb_mod, color="black")
    axes[0].plot(np.arange(2010,2020+1,1), mb_mod2, color="blue")
    axes[0].plot(np.arange(2010, 2020 + 1, 1), mb_mod3, color="green")
    axes[0].plot(np.arange(2010, 2020 + 1, 1), mb_mod4, color="orange")
    axes[0].plot(np.arange(2010,2020+1,1), mb_geod, color="red", label="Hugonnet et al. 2021")
    axes[0].fill_between(np.arange(2010,2020+1,1), mb_geod-err_geod, mb_geod+err_geod, alpha=0.3)
    axes[0].set_ylabel("Glacier-wide Mass Balance (m w.e. a$^{-1}$)")
    axes[0].set_xticks(np.arange(2010,2020+1,1))
    axes[0].legend()
    #axes[1].plot(monthly_mb.time.values, monthly_mb.values, color="black")
    #axes[1].plot(monthly_mb2.time.values, monthly_mb2.values, color="blue")
    #TSLAs
    axes[1].errorbar(tsla_obs.index, tsla_obs['SC_median'], yerr=tsla_obs['SC_stdev'], color='red', ecolor="red", marker='.')
    axes[1].plot(tsla_obs.index, tsla_obs['SC_median'], color='red', marker='.', zorder=6, label="Loibl et al. in review")
    #axes[2].errorbar(tsla_mod.index, tsla_mod['SC_med'], yerr=tsla_mod['SC_std'], color='black', ecolor="black", marker='.')
    axes[1].plot(tsla_mod.index, tsla_mod['SC_med'], marker='.',color='black')
    axes[1].plot(tsla_mod2.index, tsla_mod2['SC_med'], marker='.' ,color='blue', zorder=-1)
    axes[1].plot(tsla_mod3.index, tsla_mod3['SC_med'], marker='.' ,color='green', zorder=-1)
    axes[1].plot(tsla_mod4.index, tsla_mod4['SC_med'], marker='.' ,color='orange', zorder=5)
    axes[1].set_ylabel("Elevation (m a.s.l.)")
    axes[1].set_xlabel("Year")
    axes[1].legend()
    plt.savefig("E:/OneDrive - uibk.ac.at/PhD/PhD/Data/Slides_Preparation/output/"+'MB_TSLA_eval4.png', bbox_inches="tight")
    plt.show()

#geod mb comparison, once without lapse rate, then with lapse rate temp. then with lapse rate prec.
#etc. etc.

'''

