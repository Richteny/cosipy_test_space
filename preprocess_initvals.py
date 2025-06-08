import numpy as np
import pickle
from scipy.stats import qmc

path = "/data/scratch/richteny/for_emulator/"

# ============ Init Value Generation =============
def generate_initvals(N):
    priors = {
	'rrrfactor': (0.6218, 0.9419),
        #'rrrfactor': (0.1, 0.7),
        'albsnow': (0.887, 0.93),
        'albice': (0.118, 0.232),
        'albfirn': (0.51, 0.69),
        'albaging': (3, 12),
#        'albaging': (7.37, 24.76),
        'albdepth': (1.08, 10.753),
        'iceroughness': (1.22, 19.52),
#        'centersnow': (-3, 2) #option
    }

    param_names = list(priors.keys())
    bounds = np.array(list(priors.values()))

    sampler = qmc.LatinHypercube(d=len(priors))
    lhs_unit = sampler.random(n=N)
    lhs_scaled = qmc.scale(lhs_unit, bounds[:,0], bounds[:,1])

    initvals = [
        {param: float(lhs_scaled[i, j]) for j, param in enumerate(param_names)}
        for i in range(N)
    ]
    return initvals

## Make sure that bounds, number of chains etc. are aligned with main script!!
initvals = generate_initvals(20)
with open(path+"albaging_initvals.pkl", "wb") as f: #adjust
    pickle.dump(initvals, f)
