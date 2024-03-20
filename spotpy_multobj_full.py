import pandas as pd
from pathlib import Path
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spotpy  # Load the SPOT package into your working storage
from spotpy.parameter import Uniform
from spotpy import analyser  # Load the Plotting extension
from spotpy.objectivefunctions import rmse, mae
from COSIPY import main
from constants import *
from config import *
import random
#from dask_mpi import initialize
#initialize()

#Two options for obs dataset, Minimum TSL bodied TSLA_Abramov_filtered_full.csv or TSLA_Abramov_filtered_jaso.csv
print("TSL file:", tsl_data_file)
tsla_obs = pd.read_csv(tsl_data_file)
tsla_obs['LS_DATE'] = pd.to_datetime(tsla_obs['LS_DATE'])
time_start = "2000-01-01" #bc config starts with spinup -> need to add 1 year
time_start_dt = pd.to_datetime(time_start)
time_end_dt = pd.to_datetime(time_end)
print("Time start: ", time_start)
print("Time end: ", time_end)
tsla_obs = tsla_obs[(tsla_obs['LS_DATE'] > time_start_dt) & (tsla_obs['LS_DATE'] <= time_end_dt)]
tsla_obs.set_index('LS_DATE', inplace=True)
#Normalize standard deviation as well
if tsl_normalize:
    tsla_obs['SC_stdev'] = (tsla_obs['SC_stdev']) / (tsla_obs['glacier_DEM_max'] - tsla_obs['glacier_DEM_min'])
    #print(tsla_obs['SC_stdev'])
# Assign minimum TSLA where glacier seems fully snow-covered?
#tsla_obs['SC_median'][tsla_obs['SC_min'] <= tsla_obs['glacier_DEM_min']] = tsla_obs['SC_min']
#tsla_obs['TSL_normalized'] = (tsla_obs['SC_median'] - tsla_obs['glacier_DEM_min']) / (tsla_obs['glacier_DEM_max'] - tsla_obs['glacier_DEM_min'])
#tsla_obs['TSL_normalized'][tsla_obs['TSL_normalized'] < 0] = 0

#Set cases where minimum glacier elevation in 2nd percentile to minimum? 
#Differences large with large degree of spatial aggregation (and altitudinal)

#Load geod. MB observations
path_to_geod = "/data/scratch/richteny/Hugonnet_21_MB/"
rgi_id = "RGI60-11.00897" #"RGI60-13.18096"
rgi_region = rgi_id.split('-')[-1][:2]
geod_ref = pd.read_csv(path_to_geod+"dh_{}_rgi60_pergla_rates.csv".format(rgi_region))
geod_ref = geod_ref.loc[geod_ref['rgiid'] == rgi_id]
geod_ref = geod_ref.loc[geod_ref['period'] == "2000-01-01_2010-01-01"]
geod_ref = geod_ref[['dmdtda','err_dmdtda']]


## Load parameter list ##
param_list = pd.read_csv('/data/scratch/richteny/thesis/cosipy_test_space/cosipy_par_smpl.csv')
print(param_list.head(3))
fromlist=False
#tsl_normalize=True

class spot_setup:
    if fromlist == False:
        print("Setting parameters not from a list")
    # defining all parameters and the distribution except lrT and lrRH and lrRRR
        param = RRR_factor, alb_ice, alb_snow, alb_firn,\
                albedo_aging, albedo_depth, center_snow_transfer_function, spread_snow_transfer_function,\
                roughness_fresh_snow, roughness_ice = [
            #Uniform(low=-0.0075, high=-0.005),  # lr_temp
            #Uniform(low=0, high=0.0002),  # lr_prec
            #Uniform(low=0, high=0.1), #lr RH2 -> in percent so from 0 to 1 % no prior knowledge for this factor
            Uniform(low=0.3, high=4.), #1.235, high=1.265
            Uniform(low=0.18, high=0.4),  # alb ice
            Uniform(low=0.75, high=0.95),  # alb snow
            Uniform(low=0.4, high=0.75), #alb_firn
            Uniform(low=3, high=23), #effect of age on snow albedo (days)
            Uniform(low=3, high=20), #effect of snow depth on albedo (cm) 
            Uniform(low=-0.5, high=1.5), #snow transfer function
            Uniform(low=0.2, high=3), #spread snow transfer function
            Uniform(low=0.24, high=3.56), #fresh snow roughness
            Uniform(low=0.1, high=7.0) #ice roughness
            ]

    # Number of needed parameter iterations for parametrization and sensitivity analysis
    M = 4  # inference factor (default = 4)
    d = 2  # frequency step (default = 2)
    k = 9 #len(param)  # number of parameters

    par_iter = (1 + 4 * M ** 2 * (1 + (k - 2) * d)) * k
    print("got this far.")
    def __init__(self, obs, count="", obj_func=None):
        self.obj_func = obj_func
        self.obs = obs
        self.count = count
        if fromlist:
            print("Getting parameters from list.")
            self.params = [spotpy.parameter.List('lr_T',param_list['parlr_T'].tolist()),
                           spotpy.parameter.List('lr_RRR',param_list['parlr_RRR'].tolist()),
                           spotpy.parameter.List('lr_RH',param_list['parlr_RH'].tolist()),
                           spotpy.parameter.List('RRR_factor',param_list['parRRR_factor'].tolist()),
                           spotpy.parameter.List('alb_ice',param_list['paralb_ice'].tolist()),
                           spotpy.parameter.List('alb_snow',param_list['paralb_snow'].tolist()),
                           spotpy.parameter.List('alb_firn',param_list['paralb_firn'].tolist()),
                           spotpy.parameter.List('albedo_aging',param_list['paralbedo_aging'].tolist()),
                           spotpy.parameter.List('albedo_depth',param_list['paralbedo_depth'].tolist())]
        #else:
        #    print("Setting parameters.")
        #    self.params = lr_T, lr_RRR, lr_RH, RRR_factor, alb_ice, alb_snow, alb_firn,\
        #                      albedo_aging, albedo_depth = [
        #                  Uniform(low=-0.007, high=-0.005),
        #                  Uniform(low=0,high=0.00017),
        #                  Uniform(low=0, high=0.1),
        #                  Uniform(low=1.25, high=1.8, step=0.01), #1.235, high=1.265
        #                  Uniform(low=0.18, high=0.4,step=0.01),
        #                  Uniform(low=0.65,high=0.9,step=0.01),
        #                  Uniform(low=0.4,high=0.65,step=0.01),
        #                  Uniform(low=5,high=23,step=1),
        #                  Uniform(low=3,high=8,step=1)
        #                  ]

        print("Initialised.")

    if fromlist:
        def parameters(self):
            return spotpy.parameter.generate(self.params)

    def simulation(self, x):
        if isinstance(self.count,int):
            self.count += 1
        print("Count", self.count)
        sim_mb, sim_tsla = main(RRR_factor=x.RRR_factor,
                                alb_ice = x.alb_ice, alb_snow = x.alb_snow,alb_firn = x.alb_firn,
                                albedo_aging = x.albedo_aging, albedo_depth = x.albedo_depth,
                                center_snow_transfer_function = x.center_snow_transfer_function, 
                                spread_snow_transfer_function = x.spread_snow_transfer_function,
                                roughness_fresh_snow = x.roughness_fresh_snow, roughness_ice = x.roughness_ice,
                                count=self.count)
        sim_tsla = sim_tsla[sim_tsla['time'].isin(tsla_obs.index)]
        return (sim_tsla.Med_TSL.values, np.array([sim_mb]))

    def evaluation(self):
        obs_tsla, obs_mb = self.obs
        return (obs_tsla, obs_mb)

    def objectivefunction(self, simulation, evaluation, params=None):
        # SPOTPY expects to get one or multiple values back,
        # that define the performance of the model run
        if not self.obj_func:
            if tsl_normalize:
                print("Using normalized values.")
                eval_tsla = np.delete(evaluation[0].TSL_normalized.values, np.argwhere(np.isnan(simulation[0])))
            else:
                eval_tsla = np.delete(evaluation[0].SC_median.values, np.argwhere(np.isnan(simulation[0])))
            eval_mb = evaluation[1]['dmdtda'].values
            sigma_mb = evaluation[1]['err_dmdtda'].values
            print("MB is: ", eval_mb)
            print("Sigma MB is: ", sigma_mb)
            sigma_tsla = np.delete(evaluation[0].SC_stdev.values, np.argwhere(np.isnan(simulation[0]))) 
            sim_tsla = simulation[0][~np.isnan(simulation[0])]
            sim_mb = simulation[1][~np.isnan(simulation[1])]
            #print(sigma_tsla)
            #print(sim_tsla)
            mbe_tsla = (((eval_tsla - sim_tsla)**2) / (sigma_tsla**2)).mean()
            #rmse = (((eval_tsla - sim_tsla)**2)/(sigma_tsla**2)).mean()**.5
            print("Sim MB is: ", sim_mb)
            mbe = ((eval_mb - sim_mb)**2) / (sigma_mb**2)
            cost = -(1*mbe_tsla + 1*mbe) 
            #like1 = -(rmse(eval_tsla, sim_tsla) #set minus before func if trying to maximize, depends on algorithm
            #like2 = -mae(eval_mb,sim_mb)
            print("MBE TSLA is: ", mbe_tsla)
            print("Bias MB is: ", mbe)
            print("Full value of cost function: ", cost)
        else:
            # Way to ensure flexible spot setup class
            cost = self.obj_func(evaluation.SC_median.values, simulation.Med_TSL.values)
        return [cost]

 

def psample(obs, rep=10, count=None, dbname='cosipy_par_smpl', dbformat="csv",algorithm='mcmc'):
    #try lhs which allows for multi-objective calibration which mcmc here does not
    #set seed to make results reproducable, -> for running from list only works with mc
    np.random.seed(42)
    random.seed(42)

    M = 4
    d = 2
    k = 13
    par_iter = (1 + 4 * M ** 2 * (1 + (k -2) * d)) * k

    setup = spot_setup(obs, count=count)

    alg_selector = {'mc': spotpy.algorithms.mc, 'sceua': spotpy.algorithms.sceua, 'mcmc': spotpy.algorithms.mcmc,
                    'mle': spotpy.algorithms.mle, 'abc': spotpy.algorithms.abc, 'sa': spotpy.algorithms.sa,
                    'dds': spotpy.algorithms.dds, 'demcz': spotpy.algorithms.demcz,
                    'dream': spotpy.algorithms.dream, 'fscabc': spotpy.algorithms.fscabc,
                    'lhs': spotpy.algorithms.lhs, 'padds': spotpy.algorithms.padds,
                    'rope': spotpy.algorithms.rope}

    #save_sim = True returns error
    sampler = alg_selector[algorithm](setup, dbname=dbname, dbformat=dbformat,random_state=42,save_sim=True)
    sampler.sample(rep, nChains=5, nCr=3, eps=10e-6, convergence_limit = 1.0)

#mc to allow to read from list
mcmc = psample(obs=(tsla_obs,geod_ref), count=1, rep=10000, algorithm='dream')

#Plotting routine and most parts of script created by Phillip Schuster of HU Berlin
#Thank you Phillip!
