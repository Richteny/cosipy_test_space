import numpy as np
import arviz as az
import os

outpath = "/data/scratch/richteny/thesis/cosipy_test_space/simulations/emulator/"

idata = az.concat(
    [az.from_netcdf(f"{outpath}/stage1_chain_{i}.nc") for i in range(20) if i !=30 and i != 32 and i !=34 and i!=35 and i!=36 and i!=38 and i!=411 and i!=418 and i!=419],
    dim="chain")

idata.to_netcdf(f"{outpath}/stage1_demcz_posterior_combined.nc")

#delete individual chain output
os.chdir(outpath)
os.system("rm -r stage1_chain_*.nc")

print("Chains merged and deleted :)")
