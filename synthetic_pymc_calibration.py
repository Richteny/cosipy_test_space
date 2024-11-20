import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
import sys
import os
from COSIPY import main as runcosipy
#from constants import *
#from config import *
from cosipy.config import Config
from cosipy.constants import Constants
import pytensor
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

pytensor.config.exception_verbosity = "high"
pytensor.config.optimizer="fast_compile"
##below for pymc3
#import theano.tensor as tt
#from theano.compile.ops import as_op

def main():

    # Initiate Constants and Config
    Config()
    Constants()

    ### set up paths and constants
    main_path = "/data/scratch/richteny/thesis/cosipy_test_space/"
    os.makedirs(main_path+"simulations/", exist_ok=True)
    ## Synthetic use with simulated mass balance
    path_to_synth = "/data/scratch/richteny/thesis/cosipy_test_space/data/output/synthetic/"
    path_to_geod = "/data/scratch/richteny/Hugonnet_21_MB/"
    rgi_id = "RGI60-11.00897"
    rgi_region = rgi_id.split('-')[-1][:2]

    geod_ref = pd.read_csv(path_to_geod+"dh_{}_rgi60_pergla_rates.csv".format(rgi_region))
    geod_ref = geod_ref.loc[geod_ref['rgiid'] == rgi_id]
    geod_ref = geod_ref.loc[geod_ref['period'] == "2000-01-01_2010-01-01"]
    geod_ref = geod_ref[['dmdtda','err_dmdtda']]

    #Uncomment if 2D runs
    ### Load observations
    tsla_true_obs = pd.read_csv(Config.tsl_data_file)
    tsla_true_obs['LS_DATE'] = pd.to_datetime(tsla_true_obs['LS_DATE'])
    time_start = "2000-01-01" #config starts with spinup - need to add 1year
    time_start_dt = pd.to_datetime(time_start)
    time_end_dt = pd.to_datetime(Config.time_end)
    print("Start date:", time_start)
    print("End date:", Config.time_end)
    tsla_true_obs = tsla_true_obs.loc[(tsla_true_obs['LS_DATE'] > time_start_dt) & (tsla_true_obs['LS_DATE'] <= time_end_dt)]
    tsla_true_obs.set_index('LS_DATE', inplace=True)
    #Normalize standard deviation if necessary
    if Config.tsl_normalize:
        tsla_true_obs['SC_stdev'] = (tsla_true_obs['SC_stdev']) / (tsla_true_obs['glacier_DEM_max'] - tsla_true_obs['glacier_DEM_min'])


    # Load TSL data
    tsl_data_file_synth = "tsla_hef_cosmo_1d10m_1999_2010_horayzon_20000101-20001231_rrr-2.2_0.94_0.2_0.555_num.csv"
    print("Loading TSL file from:", tsl_data_file_synth)
    tsla_synth = pd.read_csv(path_to_synth+tsl_data_file_synth)
    tsla_synth['time'] = pd.to_datetime(tsla_synth['time'])
    #time_start = "2000-01-01" #config starts with spinup - need to add 1year
    #time_start_dt = pd.to_datetime(time_start)
    #time_end_dt = pd.to_datetime(Config.time_end)
    #print("Start date:", time_start)
    #print("End date:", Config.time_end)
    tsla_synth = tsla_synth.loc[(tsla_synth['time'] > time_start_dt) & (tsla_synth['time'] <= time_end_dt)]
    tsla_synth.set_index('time', inplace=True)
    #Normalize standard deviation if necessary
    # since we don't have stdevfrom model - set it to average of stdev from data
    #crop to same time steps as observed data to make it comparable
    tsla_synth = tsla_synth.loc[tsla_synth.index.isin(tsla_true_obs.index)]
    tsla_synth['Std_TSL'] = tsla_true_obs['SC_stdev']
    print(tsla_synth)
    
    # Load MB data
    geod_synth = xr.open_dataset(path_to_synth+"HEF_COSMO_1D10m_1999_2010_HORAYZON_20000101-20001231_RRR-2.2_0.94_0.2_0.555_num.nc")
    geod_synth = geod_synth.sel(time=slice(time_start,Config.time_end))
    geod_synth['weighted_mb'] = geod_synth['MB'] * geod_synth['N_Points'] / np.sum(geod_synth['N_Points'])
    spat_mean = geod_synth[['weighted_mb']].sum(dim=['lat','lon'])
    dfmb = spat_mean['weighted_mb'].to_dataframe()
    dfmb_ann = dfmb.resample("1Y").sum()
    geod_mb = np.nanmean(dfmb_ann['weighted_mb'].values)
    
    #use stdevs from observations
    

    #create count to store simulations 
    count = 0

    ## Define cosipy function #dvector or matrix
    #@as_op(itypes=[pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar], otypes=[pt.dscalar, pt.dvector])
    @as_op(itypes=[pt.dscalar], otypes=[pt.dscalar, pt.dvector])
    def run_cspy(rrr_factor): #, alb_ice, alb_snow, alb_firn, albedo_aging, albedo_depth):
        #params
        rrrfactor = rrr_factor
        #albice = alb_ice
        #albsnow = alb_snow
        #albfirn = alb_firn
        #albaging = albedo_aging
        #albdepth = albedo_depth
        
        #start cosipy
        modmb, modtsl = runcosipy(RRR_factor=rrrfactor, #, alb_ice=albice, alb_snow=albsnow, alb_firn=albfirn, albedo_aging=albaging,
                                  count=count)  #albedo_depth=albdepth, count=count)
        print("Calculated MB is: ", modmb)
        return np.array([modmb]), modtsl['Med_TSL'].values

    ### PYMC Model setup

    with pm.Model() as model:

        # Defining priors for the model parameters to calibrate
        rrr_factor = pm.TruncatedNormal('rrrfactor', mu=1, sigma=20, lower=0.33, upper=3)
        #alb_snow = pm.TruncatedNormal('albsnow', mu=0.89, sigma=0.5, lower=0.71, upper=0.97)
        #alb_ice = pm.TruncatedNormal('albice', mu=0.25, sigma=0.5, lower=0.1, upper=0.4)
        #alb_firn = pm.TruncatedNormal('albfirn', mu=0.55, sigma=0.5, lower=0.41, upper=0.7)
        #alb_aging = pm.TruncatedNormal('albaging', mu=14, sigma=50, lower=0.1, upper=31) #bound to positive, mu at Moelg value
        #alb_depth = pm.TruncatedNormal('albdepth', mu=6, sigma=50, lower=0.1, upper=31) #see where this comes from
        print("Set priors.")

        #Get output of COSIPY
        modmb, modtsl = run_cspy(rrr_factor) #, alb_ice, alb_snow, alb_firn, alb_aging, 
                                 #alb_depth)


        #Setup observations
        geod_data = pm.Data('geod_data', np.array([geod_mb]))
        #print(geod_ref['dmdtda'])
        #Uncomment if 2D
        tsl_data = pm.Data('tsl_data', np.array(tsla_synth['Med_TSL']))

        #print(np.array(tsla_true_obs['TSL_normalized']))
        #Expected values as deterministic RVs
        #for some reason this doesnt work !!!
        mu_mb = pm.Deterministic('mu_mb', modmb)
        #Uncomment if 2D
        mu_tsl = pm.Deterministic('mu_tsl', modtsl)

        #Likelihood (sampling distribution) of observations
        mb_obs = pm.Normal("mb_obs", mu=mu_mb, sigma=geod_ref['err_dmdtda'], observed=geod_data)
        tsl_obs = pm.Normal("tsl_obs", mu=mu_tsl, sigma=np.array(tsla_synth['Std_TSL']), observed=tsl_data, shape=mu_tsl.shape[0])

        ## Setup sampler
        #step = pm.Metropolis()
        #step = pm.Slice()
        
        # Define different initial values for each chain (6 chains in total)
        initvals = [
            {'rrrfactor': 0.6, 'albsnow': 0.75, 'albice': 0.2, 'albfirn': 0.5, 'albaging': 20, 'albdepth': 10},  # Initial values for chain 1
            {'rrrfactor': 1.0, 'albsnow': 0.79, 'albice': 0.24, 'albfirn': 0.45, 'albaging': 2, 'albdepth': 16}, # Initial values for chain 2
            {'rrrfactor': 0.4, 'albsnow': 0.72, 'albice': 0.3, 'albfirn': 0.55, 'albaging': 15, 'albdepth': 9},  # Initial values for chain 3
            {'rrrfactor': 2.6, 'albsnow': 0.8, 'albice': 0.39, 'albfirn': 0.65, 'albaging': 7, 'albdepth': 3}, # Initial values for chain 4
            {'rrrfactor': 1.6, 'albsnow': 0.9, 'albice': 0.34, 'albfirn': 0.6, 'albaging': 3, 'albdepth': 1},  # Initial values for chain 5
            {'rrrfactor': 2.0, 'albsnow': 0.95, 'albice': 0.18, 'albfirn': 0.69, 'albaging': 25, 'albdepth': 30}, # Initial values for chain 6
            ]
        #initvals = {'rrrfactor': 0.6, 'albsnow': 0.75, 'albice': 0.2, 'albfirn': 0.5, 'albaging': 20, 'albdepth': 10}
        step = pm.DEMetropolisZ()
        post = pm.sample(draws=2000, tune=200, step=step, return_inferencedata=True, chains=1, cores=1,
                         progressbar=True, discard_tuned_samples=False) #initvals=initvals

        ## testing to save samples
        post.to_netcdf(main_path+"simulations/simulations_HEF_results_MCMC.nc")

if __name__ == '__main__':
    main()
