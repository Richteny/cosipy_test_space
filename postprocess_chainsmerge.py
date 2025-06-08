import numpy as np
import arviz as az
import os

outpath = "/data/scratch/richteny/thesis/cosipy_test_space/simulations/emulator/"

idata = az.concat(
    [az.from_netcdf(f"{outpath}/chain_{i}.nc") for i in range(20) if i !=0 and i != 2 and i !=4 and i!=5 and i!=6 and i!=8 and i!=411 and i!=418 and i!=419],
    dim="chain")

idata.to_netcdf(f"{outpath}/syserr_demcz_posterior_combined.nc")

#delete individual chain output
os.chdir(outpath)
os.system("rm -r chain_*.nc")

print("Chains merged and deleted :)")
