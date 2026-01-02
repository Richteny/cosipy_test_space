import numpy as np
import arviz as az
import os

outpath = "/data/scratch/richteny/thesis/cosipy_test_space/simulations/emulator/"

idata = az.concat(
    [az.from_netcdf(f"{outpath}/debug_chain_{i}.nc") for i in range(20) if i !=10 and i !=217 and i !=219 and i!=30 and i!=214 and i!=38 and i!=411 and i!=418 and i!=419],
    dim="chain")

idata.to_netcdf(f"{outpath}/halji_debug_posterior_combined.nc")

#delete individual chain output
os.chdir(outpath)
os.system("rm -r debug_chain_*.nc")

print("Chains merged and deleted :)")
