import numpy as np
import pickle
from scipy.stats import qmc

path = "/data/scratch/richteny/for_emulator/"

#Leave a lil space to ensure that chains start in range
# ============ Init Value Generation =============
def generate_initvals(N):
    priors = {
	#'rrrfactor': (0.649, 0.946),
        'rrrfactor': (0.1, 0.6),
        #'albsnow': (0.888, 0.928),
        'albsnow': (0.88, 0.93),
        #'albice': (0.1185, 0.2232),
        'albice': (0.11, 0.25),
        #'albfirn': (0.52, 0.67),
        'albfirn': (0.46, 0.65),
        #'albaging': (5.3, 24.5),
        'albaging': (1, 6),
        #'albdepth': (1.0, 3.3),
        'albdepth': (0.9, 15),
        #'iceroughness': (1.42, 19.32),
        'iceroughness': (0.7, 19.52),
        'centersnow': (-3, 1) #option
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
with open(path+"point_initvals.pkl", "wb") as f: #adjust
    pickle.dump(initvals, f)
