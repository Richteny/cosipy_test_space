import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spotpy  # Load the SPOT package into your working storage
from spotpy import analyser  # Load the Plotting extension

res = pd.read_csv('FAST_sensitivity_parameters.csv')
res = res.drop('parRRR_factor', axis=1)
res.to_csv("mod_FAST_sensitivity_parameters.csv", index=False)

results = spotpy.analyser.load_csv_results('mod_FAST_sensitivity_parameters')

parnames = spotpy.analyser.get_parameternames(results)
print(parnames)

Si = spotpy.analyser.get_sensitivity_of_fast(results, like_index=1)
print(Si)

spotpy.analyser.plot_fast_sensitivity(results)
#plt.savefig('FAST_plot.png')
