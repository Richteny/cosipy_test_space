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
from spotpy.objectivefunctions import rmse
from COSIPY import main
from constants import *
from config import *

#Two options for obs dataset, Minimum TSL bodied TSLA_Abramov_filtered_full.csv or TSLA_Abramov_filtered_jaso.csv
print("TSL file:", tsl_data_file)
obs = pd.read_csv(tsl_data_file)
obs['LS_DATE'] = pd.to_datetime(obs['LS_DATE'])
time_start_dt = pd.to_datetime(time_start)
time_end_dt = pd.to_datetime(time_end)
print("Time start: ", time_start)
print("Time end: ", time_end)
obs = obs[(obs['LS_DATE'] > time_start_dt) & (obs['LS_DATE'] <= time_end_dt)]
obs.set_index('LS_DATE', inplace=True)

## Load variable dates from manual selection ##
var_dates_df = pd.read_csv("./data/input/Abramov/snowlines/variable_snowline_modelled_dates.csv")
var_dates_df['Variable_Dates'] = pd.to_datetime(var_dates_df['Variable_Dates'])
obs = obs[obs.index.isin(var_dates_df['Variable_Dates'])]
#count=-1

class spot_setup:
    # defining all parameters and the distribution
    param = lr_T, lr_RRR, lr_RH, RRR_factor, alb_ice, alb_snow, alb_firn,\
            albedo_aging, albedo_depth = [
        Uniform(low=-0.007, high=-0.005),  # lr_temp
        Uniform(low=0, high=0.00017),  # lr_prec
        Uniform(low=0, high=0.01), #lr RH2 -> in percent so from 0 to 1 % no prior knowledge for this factor
        Uniform(low=0.63, high=0.68), #1.235, high=1.265
        Uniform(low=0.18, high=0.4, step=0.01),  # alb ice
        Uniform(low=0.65, high=0.9, step=0.01),  # alb snow
        Uniform(low=0.4, high=0.65, step=0.01), #alb_firn
        Uniform(low=5, high=23, step=1), #effect of age on snow albedo (days)
        Uniform(low=3, high=8, step=1) #effect of snow depth on albedo (cm) 
        ]

    # Number of needed parameter iterations for parametrization and sensitivity analysis
    M = 4  # inference factor (default = 4)
    d = 2  # frequency step (default = 2)
    k = len(param)  # number of parameters

    par_iter = (1 + 4 * M ** 2 * (1 + (k - 2) * d)) * k

    def __init__(self, obs, count="", obj_func=None):
        self.obj_func = obj_func
        self.obs = obs
        self.count = count
        print("Initialised.")

    def simulation(self, x):
        if isinstance(self.count,int):
            self.count += 1
        print("Count", self.count)
        sim = main(lr_T=x.lr_T, lr_RRR=x.lr_RRR,lr_RH= x.lr_RH, RRR_factor=x.RRR_factor,
                   alb_ice = x.alb_ice, alb_snow = x.alb_snow,alb_firn = x.alb_firn,
                   albedo_aging = x.albedo_aging, albedo_depth = x.albedo_depth, count=self.count)
        sim = sim[sim['time'].isin(obs.index)]
        return sim.Med_TSL.values

    def evaluation(self):
        tsl_obs = self.obs.copy()
        return tsl_obs

    def objectivefunction(self, simulation, evaluation, params=None):
        # SPOTPY expects to get one or multiple values back,
        # that define the performance of the model run
        if not self.obj_func:
            eval = np.delete(evaluation.SC_median.values, np.argwhere(np.isnan(simulation)))
            sim = simulation[~np.isnan(simulation)]
            like = -rmse(eval, sim) #set minus before rmse if trying to maximize, depends on algorithm
            print("RMSE is: ", like)
        else:
            # Way to ensure flexible spot setup class
            like = self.obj_func(evaluation.SC_median.values, simulation.Med_TSL.values)
        return like

 

def psample(obs, rep=10, count=None, dbname='cosipy_par_smpl', dbformat="csv", interf=4, freqst=2, ngs=2,
            algorithm='mcmc', savefig=True):

    
    setup = spot_setup(obs, count=count)

    alg_selector = {'mc': spotpy.algorithms.mc, 'sceua': spotpy.algorithms.sceua, 'mcmc': spotpy.algorithms.mcmc,
                    'mle': spotpy.algorithms.mle, 'abc': spotpy.algorithms.abc, 'sa': spotpy.algorithms.sa,
                    'dds': spotpy.algorithms.dds, 'demcz': spotpy.algorithms.demcz,
                    'dream': spotpy.algorithms.dream, 'fscabc': spotpy.algorithms.fscabc,
                    'lhs': spotpy.algorithms.lhs, 'padds': spotpy.algorithms.padds,
                    'rope': spotpy.algorithms.rope}

    #save_sim = True returns error
    sampler = alg_selector[algorithm](setup, dbname=dbname, dbformat=dbformat,save_sim=True)
    sampler.sample(rep)
        
    results = sampler.getdata()
    #print("Results:", results)
    best_param = spotpy.analyser.get_best_parameterset(results)
    print("Best param:", best_param)
    par_names = spotpy.analyser.get_parameternames(best_param)
    param_zip = zip(par_names, best_param[0])
    best_param = dict(param_zip)

    bestindex, bestobjf = spotpy.analyser.get_maxlikeindex(results)  # Run with highest (lowest) RMSE
    best_model_run = results[bestindex]
    fields = [word for word in best_model_run.dtype.names if word.startswith('sim')]
    #print("Fields:\n", fields)
    # create best run data frame of TSLs
    #print("Vals:", list(list(best_model_run[fields])[0]))
    #print("Length vals:", len(list(list(best_model_run[fields])[0])))
    #print("Datatype vals:", list(list(best_model_run[fields])[0]).dtypes)
    best_simulation = pd.Series(list(list(best_model_run[fields])[0]), index= pd.to_datetime(obs.index))
    #print(best_simulation)
    # Only necessary because spot_setup.evaluation() has a datetime. Thus both need a datetime.

    fig1 = plt.figure(1, figsize=(9, 5))
    plt.plot(results['like1'])
    plt.ylabel('NS-Eff')
    plt.xlabel('Iteration')
    if savefig:
        plt.savefig(dbname + '_sampling_plot.png')
        
    fig2 = plt.figure(figsize=(16, 9))
    ax = plt.subplot(1, 1, 1)
    ax.plot(best_simulation, color='black', linestyle='solid', label='Best objf.=' + str(bestobjf))
    ax.plot(obs.SC_median, 'r.', markersize=3, label='Observation data')
    plt.xlabel('Date')
    plt.ylabel('Transient Snowline Altitude [m asl]')
    plt.legend(loc='upper right')
    if savefig:
        plt.savefig(dbname + '_best_run_plot.png')
    
    fig3 = plt.figure(figsize=(16, 9))
    ax = plt.subplot(1, 1, 1)
    q5, q25, q75, q95 = [], [], [], []
    for field in fields:
        q5.append(np.nanpercentile(results[field][-100:-1], 2.5))
        q95.append(np.nanpercentile(results[field][-100:-1], 97.5))
    ax.plot(q5, color='dimgrey', linestyle='solid')
    ax.plot(q95, color='dimgrey', linestyle='solid')
    ax.fill_between(np.arange(0, len(q5), 1), list(q5), list(q95), facecolor='dimgrey', zorder=0,
                    linewidth=0, label='parameter uncertainty')
    ax.plot(np.array(obs.SC_median), 'r.',
            label='data')  # Need to remove Timestamp from Evaluation to make comparable
    #ax.set_ylim(0, 100)
    #ax.set_xlim(0, len(obs.SC_median))
    ax.legend()
    if savefig:
        plt.savefig(dbname + '_par_uncertain_plot.png')
    
    fig4 = plt.figure(figsize=(9, 9))
    ax = plt.subplot(1, 1, 1)
    ax.scatter(obs.SC_median, best_simulation)
    ax.errorbar(obs.SC_median,best_simulation, xerr=obs.SC_stdev, fmt='o', ecolor='black', color='green')
    ax.plot([0,1],[0,1], color="blue", transform=ax.transAxes, linestyle="--")
    plt.gca().set_aspect('equal', adjustable='box') 
    ax.set_xticks(np.arange(3600, 4500, 50))
    ax.set_yticks(np.arange(3600, 4500, 50))
    ax.set_ylim(3600, 4500)
    ax.set_xlim(3600, 4500)
    ax.set_ylabel("Modelled Median TSLA [m asl]")
    ax.set_xlabel("Observed Median TSLA [m asl]")
    ax.set_title("Mod. vs. Obs. Median TSLA, Best objf.=" +str(bestobjf)) 
    if savefig:
        plt.savefig(dbname+ '_par_scatter_plot.png')
    return {'best_param': best_param, 'best_index': bestindex, 'best_model_run': best_model_run, 'best_objf': bestobjf,
            'param': spot_setup.param,'opt_iter': spot_setup.par_iter,
            'sampling_plot': fig1} #'best_simulation': best_simulation, 'best_run_plot': fig2, 'par_uncertain_plot': fig3}

#Plotting routine and most parts of script created by Phillip Schuster of HU Berlin
#Thank you Phillip!
