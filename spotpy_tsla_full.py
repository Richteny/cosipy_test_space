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

#Two options for obs dataset, Minimum TSL bodied TSLA_Abramov_filtered_full.csv or TSLA_Abramov_filtered_jaso.csv
print("TSL file:", tsl_data_file)
tsla_obs = pd.read_csv(tsl_data_file)
tsla_obs['LS_DATE'] = pd.to_datetime(tsla_obs['LS_DATE'])
time_start = "2010-01-01" #bc config starts with spinup -> need to add 1 year
time_start_dt = pd.to_datetime(time_start)
time_end_dt = pd.to_datetime(time_end)
print("Time start: ", time_start)
print("Time end: ", time_end)
tsla_obs = tsla_obs[(tsla_obs['LS_DATE'] > time_start_dt) & (tsla_obs['LS_DATE'] <= time_end_dt)]
tsla_obs.set_index('LS_DATE', inplace=True)

## Load variable dates from manual selection ##
#var_dates_df = pd.read_csv("./data/input/Abramov/snowlines/variable_snowline_modelled_dates.csv")
#var_dates_df['Variable_Dates'] = pd.to_datetime(var_dates_df['Variable_Dates'])
#obs = obs[obs.index.isin(var_dates_df['Variable_Dates'])]
#count=-1

## Load parameter list ##
#param_list = pd.read_csv('/data/scratch/richteny/thesis/cosipy_test_space/param_files/archived/2D_Wohlfahrt/cosipy_par_smpl.csv')
param_list = pd.read_csv("/data/scratch/richteny/thesis/cosipy_test_space/param_files/current/2D_Wohlfahrt_fullTSLAopt.csv")
print(param_list.head(3))
fromlist=True
#tsl_normalize=True

class spot_setup:
    if fromlist == False:
        print("Setting parameters not from a list")
    # defining all parameters and the distribution
        param = lr_T, lr_RRR, lr_RH, RRR_factor, alb_ice, alb_snow, alb_firn,\
                albedo_aging, albedo_depth = [
            Uniform(low=-0.007, high=-0.005),  # lr_temp
            Uniform(low=0, high=0.00017),  # lr_prec
            Uniform(low=0, high=0.1), #lr RH2 -> in percent so from 0 to 1 % no prior knowledge for this factor
            Uniform(low=1.1, high=2, step=0.01), #1.235, high=1.265
            Uniform(low=0.18, high=0.4, step=0.01),  # alb ice
            Uniform(low=0.65, high=0.9, step=0.01),  # alb snow
            Uniform(low=0.4, high=0.65, step=0.01), #alb_firn
            Uniform(low=5, high=23, step=1), #effect of age on snow albedo (days)
            Uniform(low=3, high=8, step=1) #effect of snow depth on albedo (cm) 
            ]

    #Number of needed parameter iterations for parametrization and sensitivity analysis
    M = 4  # inference factor (default = 4)
    d = 2  # frequency step (default = 2)
    k = 9 #len(param)  # number of parameters

    par_iter = (1 + 4 * M ** 2 * (1 + (k - 2) * d)) * k

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
        #    self.params = [spotpy.parameter.Uniform('lr_T',[-0.007,-0.005]),
        #                   spotpy.parameter.Uniform('lr_RRR',[0, 0.00017]),
        #                   spotpy.parameter.Uniform('lr_RH',[0, 0.1]),
        #                   spotpy.parameter.Uniform('RRR_factor',[1.25,1.8,0.01]), #1.235, high=1.265
        #                   spotpy.parameter.Uniform('alb_ice',[0.18,0.4,0.01]),
        #                   spotpy.parameter.Uniform('alb_snow',[0.65,0.9,0.01]),
        #                   spotpy.parameter.Uniform('alb_firn',[0.4,0.65,0.01]),
        #                   spotpy.parameter.Uniform('albedo_aging',[5,23,1]),
        #                   spotpy.parameter.Uniform('albedo_depth',[3,8,1])]

        print("Initialised.")

    if fromlist:
        def parameters(self):
            return spotpy.parameter.generate(self.params)


    def simulation(self, x):
        if isinstance(self.count,int):
            self.count += 1
        print("Count", self.count)
        sim_tsla, sim_mb = main(lr_T=x.lr_T, lr_RRR=x.lr_RRR,lr_RH= x.lr_RH, RRR_factor=x.RRR_factor,
                                alb_ice = x.alb_ice, alb_snow = x.alb_snow,alb_firn = x.alb_firn,
                                albedo_aging = x.albedo_aging, albedo_depth = x.albedo_depth, count=self.count)
        sim_tsla = sim_tsla[sim_tsla['time'].isin(tsla_obs.index)]
        return (sim_tsla.Med_TSL.values, np.array([sim_mb]))

    def evaluation(self):
        tsl_obs = self.obs.copy()
        return tsl_obs

    def objectivefunction(self, simulation, evaluation, params=None):
        # SPOTPY expects to get one or multiple values back,
        # that define the performance of the model run
        if not self.obj_func:
            if tsl_normalize:
                print("Using normalized values.")
                eval = np.delete(evaluation.TSL_normalized.values, np.argwhere(np.isnan(simulation[0])))
            else:
                eval = np.delete(evaluation.SC_median.values, np.argwhere(np.isnan(simulation[0])))
            sim_tsla = simulation[0][~np.isnan(simulation[0])]
            sim_mb = simulation[1][~np.isnan(simulation[1])]
            #print("Simulations")
            #print(sim_tsla)
            #print("Eval.")
            #print(eval)
            like = -rmse(eval, sim_tsla) #set minus before rmse if trying to maximize, depends on algorithm
            like2 = -mae(eval,sim_tsla)
            print("RMSE is: ", like)
            print("MAE is: ", like2)
        else:
            # Way to ensure flexible spot setup class
            like = self.obj_func(evaluation.SC_median.values, simulation.Med_TSL.values)
        return [like]

 

def psample(obs, rep=10, count=None, dbname='cosipy_par_smpl', dbformat="csv", algorithm='mcmc'):
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
    sampler.sample(rep)

mcmc = psample(obs=tsla_obs, count=None, rep=500, algorithm='mc')

#Plotting routine and most parts of script created by Phillip Schuster of HU Berlin
#Thank you Phillip!
