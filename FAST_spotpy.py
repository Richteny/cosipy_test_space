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

obs = None
fromlist = False

class spot_setup:
    # defining all parameters and the distribution
    print("Setting parameters.")
    lr_T = Uniform(low=-0.007, high=-0.005)
    lr_RRR = Uniform(low=0, high=0.001)
    lr_RH = Uniform(low=0, high=0.2)
    RRR_factor = Uniform(low=1, high=2) #1.235, high=1.265
    alb_ice  = Uniform(low= 0.18, high=0.4, step=0.01)
    alb_snow = Uniform(low=0.65, high=0.9, step=0.01)
    alb_firn = Uniform(low=0.4, high=0.65, step=0.01)
    albedo_aging = Uniform(low=2, high=23, step=0.5)
    albedo_depth = Uniform(low=2, high=9, step=0.5)
    center_snow_transfer_function = Uniform(low=0, high=2, step=0.1)
    roughness_fresh_snow = Uniform(low=0.15, high=0.3, step=0.01)
    roughness_ice = Uniform(low=0.7, high=0.27, step=0.01)
    roughness_firn = Uniform(low=2, high=6, step=0.01)

    def __init__(self, obj_func=None):
        self.obj_func = obj_func
        self.trueObs = []
        self.count = 1
        print("Initialised.")

    def simulation(self, x):
        print("Count", self.count)
        sim = main(lr_T=x.lr_T, lr_RRR=x.lr_RRR,lr_RH= x.lr_RH, RRR_factor=x.RRR_factor,
                   alb_ice = x.alb_ice, alb_snow = x.alb_snow,alb_firn = x.alb_firn,
                   albedo_aging = x.albedo_aging, albedo_depth = x.albedo_depth, center_snow_transfer_function = x.center_snow_transfer_function,
                   roughness_fresh_snow = x.roughness_fresh_snow, roughness_ice = x.roughness_ice, roughness_firn = x.roughness_firn, count=self.count)
        self.count = self.count + 1
        return sim

    def evaluation(self):
        return self.trueObs

    def objectivefunction(self, simulation, evaluation, params=None):
        like = 1
        return like

 

def psample(dbname='FAST_sensitivity_parameters', dbformat="csv"):
    #set seed to make results reproducable, -> for running from list only works with mc
    np.random.seed(42)
    random.seed(42)
    
    M = 4
    d = 2
    k = 13
    par_iter = (1 + 4 * M ** 2 * (1 + (k -2) * d)) * k
    
    rep= 10000
    setup = spot_setup()
    sampler = spotpy.algorithms.fast(setup, dbname=dbname, dbformat=dbformat,random_state=42,db_precision=np.float32)
    sampler.sample(rep)
        
fast = psample()
