import numpy as np
import pickle
from scipy.stats import qmc

path = "/data/scratch/richteny/for_emulator/"

# ============ Init Value Generation =============
def generate_initvals(N):
    priors = {
	'rrrfactor': (0.5738, 1.29),
        'albsnow': (0.887, 0.93),
        'albice': (0.115, 0.233),
        'albfirn': (0.5, 0.692),
        'albaging': (2, 25),
        'albdepth': (1, 14),
        'iceroughness': (0.92, 20),
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
with open(path+"initvals.pkl", "wb") as f:
    pickle.dump(initvals, f)
