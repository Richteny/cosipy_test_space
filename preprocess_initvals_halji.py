import numpy as np
import pickle
from scipy.stats import qmc

path = "/data/scratch/richteny/for_emulator/Halji/"

#Leave a lil space to ensure that chains start in range
# ============ Init Value Generation =============
def generate_initvals(N):
    priors = {
        'rrrfactor': (0.57, 0.86),
        "albsnow": (0.83, 0.925),
        "albice": (0.13, 0.27),
        "albfirn": (0.46, 0.68),
        "albaging": (3, 23),
        "albdepth": (1.0, 12),
        "iceroughness": (0.7, 19.5),
        "lwinfactor": (0.95011, 1.05),
        "wsfactor": (0.75, 2.5)
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
with open(path+"halji_point_initvals.pkl", "wb") as f: #adjust
    pickle.dump(initvals, f)
