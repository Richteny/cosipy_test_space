import pandas as pd
from pathlib import Path
import sys
import os
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import spotpy  # Load the SPOT package into your working storage
from spotpy.parameter import Uniform
from spotpy import analyser  # Load the Plotting extension
from spotpy.objectivefunctions import rmse, mae
from cosipy.config import Config
from cosipy.constants import Constants
from COSIPY import main as runcosipy
#from constants import * #old files and importing with * not good
#from config import *
import random

#initiate config and constants
Config()
Constants()

# Set up MB data
path_to_geod = "/data/scratch/richteny/Hugonnet_21_MB/"
rgi_id = "RGI60-11.00897"
rgi_region = rgi_id.split('-')[-1][:2]

geod_ref = pd.read_csv(path_to_geod+"dh_{}_rgi60_pergla_rates.csv".format(rgi_region))
geod_ref = geod_ref.loc[geod_ref['rgiid'] == rgi_id]
geod_ref = geod_ref.loc[geod_ref['period'] == "2000-01-01_2010-01-01"]
geod_ref = geod_ref[['dmdtda','err_dmdtda']]

# Load TSL Data
print("Loading TSL file from:", Config.tsl_data_file)
tsla_obs = pd.read_csv(Config.tsl_data_file)
tsla_obs['LS_DATE'] = pd.to_datetime(tsla_obs['LS_DATE'])
time_start = "2000-01-01" #config starts with spinup - need to add 1 year
time_start_dt = pd.to_datetime(time_start)
time_end_dt = pd.to_datetime(Config.time_end)
print("Start date:", time_start)
print("End date:", Config.time_end)
tsla_obs = tsla_obs.loc[(tsla_obs['LS_DATE'] > time_start_dt) & (tsla_obs['LS_DATE'] <= time_end_dt)]
tsla_obs.set_index('LS_DATE', inplace=True)
#normalize standard deviation if necessary
if Config.tsl_normalize:
    tsla_obs['SC_stdev'] = (tsla_obs['SC_stdev']) / (tsla_obs['glacier_DEM_max'] - tsla_obs['glacier_DEM_min'])

obs = None

# Number of iterations
#N=(1+(4*M**2)*(1+(k−2)*d))*k

class spot_setup:
    # defining all parameters and the distribution
    print("Setting parameters.")
    param = RRR_factor, alb_ice, alb_snow, alb_firn, albedo_aging, albedo_depth = [
            #aging_factor_roughness, roughness_fresh_snow, roughness_ice = [
        Uniform(low=0.33, high=3), #1.235, high=1.265
        Uniform(low= 0.1, high=0.4),
        Uniform(low=0.71, high=0.98),
        Uniform(low=0.41, high=0.7),
        Uniform(low=0.1, high=31),
        Uniform(low=0.1, high=31),
        #Uniform(low=0.005, high=0.0026+0.0026),
        #Uniform(low=0.2, high=3.56),
        #Uniform(low=0.1, high=7.0)]
        #roughness_firn = Uniform(low=2, high=6) not used here but set to max. ice value because of equation
    def __init__(self, obs, count="", obj_func=None):
        self.obj_func = obj_func
        self.obs = obs
        self.trueObs = []
        self.count = 1
        print("Initialised.")

    def simulation(self, x):
        print("Count", self.count)
        sim_mb, sim_tsla = runcosipy(RRR_factor=x.RRR_factor, alb_ice = x.alb_ice, alb_snow = x.alb_snow, alb_firn = x.alb_firn,
                   albedo_aging = x.albedo_aging, albedo_depth = x.albedo_depth, #aging_factor_roughness = x.aging_factor_roughness,
                   count=self.count) #roughness_fresh_snow = x.roughness_fresh_snow, roughness_ice = x.roughness_ice, count=self.count)
        sim_tsla = sim_tsla[sim_tsla['time'].isin(tsla_obs.index)]
        return (np.array([sim_mb]), sim_tsla['Med_TSL'].values)

    def evaluation(self):
        obs_mb, obs_tsla = self.obs
        return (obs_mb, obs_tsla)

    def objectivefunction(self, simulation, evaluation, params=None):
        if not self.obj_func:
            print(evaluation[1])
            if Config.tsl_normalize:
                eval_tsla = np.delete(evaluation[1]['TSL_normalized'].values, np.argwhere(np.isnan(simulation[1])))
            else:
                eval_tsla = np.delete(evaluation[1]['SC_median'].values, np.argwhere(np.isnan(simulation[1])))
            eval_mb = evaluation[0]['dmdtda'].values
            sigma_mb = evaluation[0]['err_dmdtda'].values
            sigma_tsla = np.delete(evaluation[1]['SC_stdev'].values, np.argwhere(np.isnan(simulation[1])))
            sim_tsla = simulation[1][~np.isnan(simulation[1])]
            sim_mb = simulation[0][~np.isnan(simulation[0])]
           
            #calculate loglikelihood of both
            # Equation the same as the one below
            #loglike_mb = np.log(( 1 / np.sqrt( (2*np.pi* (sigma_mb**2) ) ) ) * np.exp( (-1* ( (eval_mb-sim_mb) **2 ) / (2*sigma_mb**2))))
            loglike_mb = -0.5 * (np.log(2 * np.pi * sigma_mb**2) + ( ((eval_mb-sim_mb)**2) / sigma_mb**2))
            loglike_tsla = -0.5 * np.sum(np.log(2 * np.pi * sigma_tsla**2) + ( ((eval_tsla-sim_tsla)**2) / sigma_tsla**2))
            #equation below works for constant sigma
            #loglike_tsla = np.log(( 1 / np.sqrt( (2*np.pi* (sigma_tsla**2) ) ) ) * np.exp( (-1* ( (eval_tsla-sim_tsla) **2 ) / (2*sigma_tsla**2))))
            like = loglike_mb + loglike_tsla
        return like

 

def psample(obs, count=None):
    #set seed to make results reproducable, -> for running from list only works with mc
    np.random.seed(42)
    random.seed(42)
    
    #M = 4
    #d = 2
    #k = 9
    #par_iter = (1 + 4 * M ** 2 * (1 + (k -2) * d)) * k
    
    rep= 3000
    count=count
    name = "LHS_surrogate_parameters"
    setup = spot_setup(obs, count=count)
    sampler = spotpy.algorithms.lhs(setup, dbname=name, dbformat='csv', db_precision=np.float32, random_state=42, save_sim=True)
    sampler.sample(rep)
        
fast = psample(obs=(geod_ref, tsla_obs), count=1)
