## import of necessary packages
import pandas as pd
from pathlib import Path
import sys
import spotpy  # Load the SPOT package into your working storage
import numpy as np
from spotpy import analyser  # Load the Plotting extension
from spotpy_full import *
import matplotlib.pyplot as plt 
from COSIPY import main 


obs = pd.read_csv("./data/input/Abramov/snowlines/TSLA_Abramov_filtered_full.csv")
obs['LS_DATE'] = pd.to_datetime(obs['LS_DATE'])
print("Time start: ", NAMELIST['time_start'])
print("Time end: ", NAMELIST['time_end'])
obs = obs[(obs['LS_DATE'] > NAMELIST['time_start']) & (obs['LS_DATE'] <= NAMELIST['time_end'])]
obs.set_index('LS_DATE', inplace=True)

best_summary = psample(obs=obs, rep=2)

## Sensitivity Analysis
spotpy_setup = spot_setup(obs)  # only once

sampler = spotpy.algorithms.fast(spotpy_setup, dbname='COSIPY_FAST', dbformat="csv")
sampler.sample(10)  # minimum 60 to run through,
# ideal number of iterations: spot_setup.par_iter, immer wieder einzelne Zeilen "out of bounds"
results = sampler.getdata()
analyser.plot_fast_sensitivity(results, number_of_sensitiv_pars=5, fig_name="FAST_sensitivity_COSIPY.png")

SI = spotpy.analyser.get_sensitivity_of_fast(results)  # Sensitivity indexes as dict

'''
def load_psample(path, max=True, cond1={'par1': 'parTT_rain', 'operator': 'lessthan', 'par2': 'parTT_snow'}):
    results = spotpy.analyser.load_csv_results(path)
    trues = results[(results['parTT_snow'] < results['parTT_rain']) & (results['parCFMAX_ice'] > results['parCFMAX_snow'])]

    trues = results[(results[cond1.get()])]


    likes = trues['like1']
    if max:
        obj_val = np.nanmax(likes)
    else:
        obj_val = np.nanmin(likes)

    index = np.where(likes == obj_val)
    best_param = trues[index]
    best_param_values = spotpy.analyser.get_parameters(trues[index])[0]
    par_names = spotpy.analyser.get_parameternames(trues)
    param_zip = zip(par_names, best_param_values)
    best_param = dict(param_zip)

    return [best_param, obj_val]

def filt(left, operator, right):
    return operator(left, right)

def lessthan(left, right):
    return filt(left, (lambda a, b: a < b), right)

def greaterthan(left, right):
    return filt(left, (lambda a, b: a > b), right)

cond1={'par1': 'parTT_rain', 'operator': '<', 'par2': 'parTT_snow'}
cond1.get('par1')

if cond1.get('operator') == '<':
    zero = results[lessthan(results[cond1.get('par1')], results[cond1.get('par2')])]
elif cond1.get('operator') == '>':
    zero = results[greaterthan(results[cond1.get('par1')], results[cond1.get('par2')])]


one = results[lessthan(results['parTT_snow'], results['parTT_rain'])]

two = results[filt(results['parTT_snow'],(lambda a, b: a<b), results['parTT_rain'])]

three = results[results['parTT_snow'] < results['parTT_rain']]

'''
