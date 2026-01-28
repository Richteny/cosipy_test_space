import numpy as np
import arviz as az
import os

outpath = "/data/scratch/richteny/thesis/cosipy_test_space/simulations/emulator/"

idata = az.concat(
    [az.from_netcdf(f"{outpath}/debug2_chain_{i}.nc") for i in range(20) if i !=5 and i !=7 and i !=9 and i!=20 and i!=0 and i!=215 and i!=216 and i!=217 and i!=419],
    dim="chain")

idata.to_netcdf(f"{outpath}/halji_debug2_posterior_combined.nc")

#delete individual chain output
os.chdir(outpath)
os.system("rm -r debug2_chain_*.nc")

print("Chains merged and deleted :)")
