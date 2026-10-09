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
from point_COSIPY import main as runcosipy
#from constants import * #old files and importing with * not good
#from config import *
import gc
import random
import time

#initiate config and constants
Config()
Constants()

# Set up data
observed = pd.read_csv("/data/scratch/richteny/thesis/cosipy_test_space/data/input/HEF/cosipy_validation_upper_station.csv",
                       parse_dates=True, index_col="time")
unc_lwo = 15 #Wm-2
unc_alb = 0.05 
unc_sfc = 0.12
obs = None

# Number of iterations
#N=(1+(4*M**2)*(1+(k−2)*d))*k

class spot_setup:
    # defining all parameters and the distribution
    print("Setting parameters.")
    param = RRR_factor, alb_ice, alb_snow, alb_firn, albedo_aging, albedo_depth, roughness_ice = [ #center_snow_transfer_function = [
            #aging_factor_roughness, roughness_fresh_snow, roughness_ice = [
        Uniform(low=np.log(0.3), high=np.log(0.95)), #1.235, high=1.265
        Uniform(low=0.115, high=0.263),
        Uniform(low=0.887, high=0.94),
        Uniform(low=0.46, high=0.692),
        Uniform(low=1, high=25),
        Uniform(low=0.9, high=14.2),
        Uniform(low=0.7, high=20)]
        #Uniform(low=-1.0, high=2.5)]
        #Uniform(low=0.005, high=0.0026+0.0026),
        #Uniform(low=0.2, high=3.56),
        #Uniform(low=0.1, high=7.0)]
        #roughness_firn = Uniform(low=2, high=6) not used here but set to max. ice value because of equation
    def __init__(self, obs, count="", obj_func=None):
        self.obj_func = obj_func
        self.obs = obs
        self.trueObs = []
        self.count = count
        print("Initialised.")

    def simulation(self, x):
        if isinstance(self.count,int):
            self.count += 1
        print("Count", self.count)
        sim_lwo, sim_alb, sim_sfc = runcosipy(RRR_factor=np.exp(x.RRR_factor), alb_ice = x.alb_ice, alb_snow = x.alb_snow, alb_firn = x.alb_firn,
                   albedo_aging = x.albedo_aging, albedo_depth = x.albedo_depth, roughness_ice = x.roughness_ice, #center_snow_transfer_function = x.center_snow_transfer_function, #aging_factor_roughness = x.aging_factor_roughness,
                   count=self.count) #roughness_fresh_snow = x.roughness_fresh_snow, roughness_ice = x.roughness_ice, count=self.count)
        gc.collect()  # Force garbage collection to avoid memory leaks

        #sim_tsla = sim_tsla[sim_tsla['time'].isin(tsla_obs.index)]
        #put a pause - waits n seconds after runcosipy has finished giving SLURM time to clean up, maybe that fixes problems
        time.sleep(3)
        return (sim_lwo, sim_alb, sim_sfc)

    def evaluation(self):
        obs_lwo, obs_alb, obs_sfc = self.obs
        return (obs_lwo, obs_alb, obs_sfc)

    def objectivefunction(self, simulation, evaluation, params=None):
        if not self.obj_func:
            try:
                eval_lwo = evaluation[0].values
                sigma_lwo = unc_lwo
                eval_alb = evaluation[1].values
                sigma_alb = unc_alb
                eval_sfc = evaluation[2].values
                sigma_sfc = unc_sfc

                sim_lwo = simulation[0]
                sim_alb = simulation[1]
                sim_sfc = simulation[2]

                #calculate loglikelihood of both
                # Equation the same as the one below
                #loglike_mb = np.log(( 1 / np.sqrt( (2*np.pi* (sigma_mb**2) ) ) ) * np.exp( (-1* ( (eval_mb-sim_mb) **2 ) / (2*sigma_mb**2))))
                loglike_lwo = -0.5 * np.sum(np.log(2 * np.pi * sigma_lwo**2) + ( ((eval_lwo-sim_lwo)**2) / sigma_lwo**2))
                loglike_alb = -0.5 * np.sum(np.log(2 * np.pi * sigma_alb**2) + ( ((eval_alb-sim_alb)**2) / sigma_alb**2))
                loglike_sfc = -0.5 * np.sum(np.log(2 * np.pi * sigma_sfc**2) + ( ((eval_sfc-sim_sfc)**2) / sigma_sfc**2))
                mean_loglike_lwo = loglike_lwo / len(sim_lwo)
                mean_loglike_alb = loglike_alb / len(sim_alb)
                mean_loglike_sfc = loglike_sfc / len(sim_sfc)
                #equation below works for constant sigma
                #loglike_tsla = np.log(( 1 / np.sqrt( (2*np.pi* (sigma_tsla**2) ) ) ) * np.exp( (-1* ( (eval_tsla-sim_tsla) **2 ) / (2*sigma_tsla**2))))
                like = mean_loglike_lwo + mean_loglike_alb + mean_loglike_sfc
            except:
                like = 999
        return like

 

def psample(obs, count=None):
    #set seed to make results reproducable, -> for running from list only works with mc
    np.random.seed(42)
    random.seed(42)
    
    #M = 4
    #d = 2
    #k = 9
    #par_iter = (1 + 4 * M ** 2 * (1 + (k -2) * d)) * k
    
    rep= 10000
    count=count
    name = "pointLHS-parameters_full"
    setup = spot_setup(obs, count=count)
    sampler = spotpy.algorithms.lhs(setup, dbname=name, dbformat='csv', db_precision=np.float64, random_state=42, save_sim=True)
    sampler.sample(rep)
        
fast = psample(obs=(observed['LWout'], observed['albedo'], observed['SR50']), count=1)
