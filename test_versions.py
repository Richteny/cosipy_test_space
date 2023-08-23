import yaml
import numpy as np
import pandas as pd
import xarray as xr
import distributed
import dask
import dask_jobqueue
import numba
import mpi4py
print("yaml version: ", yaml.__version__)
print("numpy version: ", np.__version__)
print("pandas version: ", pd.__version__)
print("xarray version: ", xr.__version__)
print("distributed version: ", distributed.__version__)
print("dask version: ", dask.__version__)
print("dask jobqueue version: ", dask_jobqueue.__version__)
print("numba version: ", numba.__version__)
print("mpi version: ", mpi4py.__version__)
