import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import pandas as pd
import sys
import os
from COSIPY import main as runcosipy
from constants import *
from config import *
#import pytensor
#import pytensor.tensor as pt
#from pytensor.compile.ops import as_op

##below for pymc3
import theano.tensor as tt
from theano.compile.ops import as_op

def main():

    ### set up paths and constants
    main_path = "/data/scratch/richteny/thesis/cosipy_test_space/"
    os.makedirs(main_path+"simulations/", exist_ok=True)
    path_to_geod = "/data/scratch/richteny/Hugonnet_21_MB/"
    rgi_id = "RGI60-11.00897"
    rgi_region = rgi_id.split('-')[-1][:2]

    ### Load observations
    # Load TSL data
    print("Loading TSL file from:", tsl_data_file)
    tsla_obs = pd.read_csv(tsl_data_file)
    tsla_obs['LS_DATE'] = pd.to_datetime(tsla_obs['LS_DATE'])
    time_start = "2000-01-01" #config starts with spinup - need to add 1year
    time_start_dt = pd.to_datetime(time_start)
    time_end_dt = pd.to_datetime(time_end)
    print("Start date:", time_start)
    print("End date:", time_end)
    tsla_obs = tsla_obs.loc[(tsla_obs['LS_DATE'] > time_start_dt) & (tsla_obs['LS_DATE'] <= time_end_dt)]
    tsla_obs.set_index('LS_DATE', inplace=True)
    #Normalize standard deviation if necessary
    if tsl_normalize:
        tsla_obs['SC_stdev'] = (tsla_obs['SC_stdev']) / (tsla_obs['glacier_DEM_max'] - tsla_obs['glacier_DEM_min'])

    # Load MB data
    geod_ref = pd.read_csv(path_to_geod+"dh_{}_rgi60_pergla_rates.csv".format(rgi_region))
    geod_ref = geod_ref.loc[geod_ref['rgiid'] == rgi_id]
    geod_ref = geod_ref.loc[geod_ref['period'] == "2000-01-01_2010-01-01"]
    geod_ref = geod_ref[['dmdtda', 'err_dmdtda']]

    ## Define cosipy function
    @as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar], otypes=[tt.dscalar, tt.dvector])
    def run_cspy(alb_ice, alb_snow, alb_firn, albedo_aging, albedo_depth, center_snow_transfer_function):
        #params
        albice = alb_ice
        albsnow = alb_snow
        albfirn = alb_firn
        albaging = albedo_aging
        albdepth = albedo_depth
        centersnow = center_snow_transfer_function

        #start cosipy
        modmb, modtsl = runcosipy(albice, albsnow, albfirn, albaging, albdepth,
                                  centersnow)

        return modmb, modtsl

    ### PYMC Model setup

    with pm.Model() as model:

        # Defining priors for the model parameters to calibrate
        albsnow = pm.TruncatedNormal('albsnow', mu=0.89, sigma=0.1, lower=0.71, upper=0.97)
        albice = pm.TruncatedNormal('albice', mu=0.25, sigma=0.1, lower=0.1, upper=0.4)
        albfirn = pm.TruncatedNormal('albfirn', mu=0.55, sigma=0.1, lower=0.41, upper=0.7)
        albaging = pm.TruncatedNormal('albaging', mu=6+3, sigma=10, lower=0.1) #bound to positive, mu at Moelg value
        albdepth = pm.TruncatedNormal('albdepth', mu=3, sigma=8, lower=0.1) #see where this comes from
        centersnow = pm.TruncatedNormal('centersnow', mu=0., sigma=0.2, lower=-3, upper=3) #reasons
        #tau = pm.DiscreteUniform("tau", lower=0, upper=10)
        print("Set priors.")

        #Get output of COSIPY
        modmb, modtsl = run_cspy(albice, albsnow, albfirn, albaging, 
                                 albdepth, centersnow)


        print("Modelled MB is:", modmb)
        #print("Ran model.")
        #Setup observations
        geod_data = pm.Data('geod_data', geod_ref['dmdtda'])
        #tsl_data = pm.Data('tsl_data', tsla_obs['SC_median'])

        #Expected values as deterministic RVs
        #for some reason this doesnt work !!!
        mu_mb = pm.Deterministic('mu_mb', modmb)
        #mu_tsl = pm.Deterministic('mu_tsl', modtsl)

        #Likelihood (sampling distribution) of observations
        mb_obs = pm.Normal("mb_obs", mu=modmb, sigma=geod_ref['err_dmdtda'], observed=geod_data)
        #tsl_obs = pm.Normal("tsl_obs", mu=mu_tsl, sigma=tsla_obs['SC_stdev'], observed=tsl_data)

        ## Setup sampler
        step = pm.Metropolis()
        post = pm.sample(draws=1, tune=1, step=step, return_inferencedata=True, chains=1, cores=10, progressbar=True,)

        ## testing to save samples
        post.to_netcdf(main_path+"simulations/simulations_HEF_results_MCMC.nc")

if __name__ == '__main__':
    main()
