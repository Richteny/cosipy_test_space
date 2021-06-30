#issue not with resampling, all values there but probably with drop=True in ds selection
import pandas as pd
import os
import numpy as np
import spotpy
from spotpy.parameter import Uniform
from spotpy.objectivefunctions import rmse
from spotpy import analyser
from COSIPY import main
from constants import *
from config import *

obs = pd.read_csv("./data/input/Abramov/snowlines/TSLA_Abramov_filtered_full.csv")
obs['LS_DATE'] = pd.to_datetime(obs['LS_DATE'])
time_start_dt = pd.to_datetime(time_start)
time_end_dt = pd.to_datetime(time_end)
print("Time start: ", time_start)
print("Time end: ", time_end)
#just for test run
obs = obs[(obs['LS_DATE'] > time_start_dt) & (obs['LS_DATE'] <= time_end_dt)]
obs.set_index('LS_DATE', inplace=True)

count=0

class spot_setup:

    param = lr_T, lr_RRR, RRR_factor, alb_ice, alb_snow, alb_firn = [
        Uniform(low=-0.0064, high=-0.0057), #lrT
        Uniform(low=0, high=0.00013), #lrRRR
        Uniform(low=0.63, high=0.68), #RRR_factor
        Uniform(low=0.2, high=0.32), #alb_ice -> alb ice can never be below alb snow, implement that somehow
        Uniform(low=0.75, high=0.9), #alb_snow
        Uniform(low=0.4, high=0.6) #alb_firn
    ]

    #M = 4 #inference factor
    #d =2  #frequency step
    #k = len(param) #number of params

    def __init__(self,obs, obj_func=None):
        self.obj_func = obj_func
        self.obs = obs

    def simulation(self, x):
        sim = main(lr_T=x.lr_T, lr_RRR=x.lr_RRR, RRR_factor=x.RRR_factor,
                   alb_ice = x.alb_ice, alb_snow = x.alb_snow,
                   alb_firn = x.alb_firn, count=count)
        print("Got simulation.")
        sim = sim[sim['time'].isin(obs.index)]
        return sim.Med_TSL.values

    def evaluation(self):
        tsl_obs = self.obs.copy()
        print("Got evaluation dataset.")
        return tsl_obs

    def objectivefunction(self, simulation, evaluation, params=None):
        if not self.obj_func:
            eval = np.delete(evaluation.SC_median.values, np.argwhere(np.isnan(simulation)))
            sim = simulation[~np.isnan(simulation)]
            print("Starting RMSE calculation.")
            print(len(sim))
            print(len(eval))
            like = -rmse(eval, sim)
            print("Calculated RMSE")
        else:
            evaluation = evaluation[evaluation.index.isin(simulation['sims']['time'])]
            sims = simulation['sims'][simulation['sims']['time'].isin(evaluation.index)]
            like = self.obj_func(evaluation.SC_median.values, sims.Med_TSL.values)
        return like

    #def save(self, objectivefunctions, parameter, simulations):
    #    line=str(objectivefunctions)+','+str(parameter).strip('[]')+','+str(simulations).strip('[]')+'\n'
    #    self.database.write(line)

rep = 3
spot_setup_class = spot_setup(obs)
print("Created setup.")
sampler = spotpy.algorithms.mc(spot_setup_class, dbname='no_simulations', dbformat='csv', save_sim=True)
print("Starting sampling.")
sampler.sample(rep)
print("Finished sampling.")
