from cosipy.config import Config
from cosipy.constants import Constants
from cosipy.modules.evaluation import evaluate, resample_output, create_tsl_df, eval_tsl, resample_by_hand
from COSIPY import prereq_res, construct_resampled_ds
from numba import njit
import xarray as xr
import pandas as pd
import numpy as np
import pathlib
from datetime import datetime

path_to_data = "/data/scratch/richteny/thesis/cosipy_test_space/data/output/LHS/"
outpath = "/data/scratch/richteny/thesis/io/data/output/nn_data/full_snowlines/"

for fp in pathlib.Path(path_to_data).glob('*.nc'):
    print(fp)

    name = "tsla_" + str(fp.stem).lower() + ".csv"
    print(name)
    times = datetime.now()
    ds = xr.open_dataset(fp)

    dates,clean_day_vals,secs,holder = prereq_res(ds)
    resampled_array = resample_by_hand(holder, ds.SNOWHEIGHT.values, secs, clean_day_vals)
    resampled_out = construct_resampled_ds(ds,resampled_array,dates.values)

    print("Time required for resampling of output: ", datetime.now()-times)
    #Need HGT values as 2D, ensured with following line of code.
    resampled_out['HGT'] = (('lat','lon'), ds['HGT'].data)
    resampled_out['MASK'] = (('lat','lon'), ds['MASK'].data)

    tsl_out = create_tsl_df(resampled_out, Config.min_snowheight, Config.tsl_method, Config.tsl_normalize)
    tsl_out.to_csv(outpath+name)

