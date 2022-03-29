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

count=1999
best_summary = psample(obs=obs,count=count, rep=2)

