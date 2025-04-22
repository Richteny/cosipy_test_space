import numpy as np
import arviz as az
import os

outpath = "/data/scratch/richteny/thesis/cosipy_test_space/simulations/emulator/"

idata = az.concat(
    [az.from_netcdf(f"{outpath}/chain_{i}.nc") for i in range(20)],
    dim="chain"
)
idata.to_netcdf(f"{outpath}/posterior_combined.nc")

#delete individual chain output
os.chdir(outpath)
os.system("rm -r chain_*.nc")

print("Chains merged and deleted :)")
