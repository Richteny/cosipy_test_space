import sys
import os
import xarray as xr
import numpy as np
from itertools import product
import richdem as rd
import fiona
import os
import matplotlib.pyplot as plt
WRF=False

#from horayzon.domain import curved_grid

os.chdir("/home/niki/Dokumente/cosipy_test_space/utilities/createStatic/")
static_folder = '../../data/static/HEF/'

sys.path.append("../..")
from utilities.aws2cosipy.crop_file_to_glacier import crop_file_to_glacier

static_raw = xr.open_dataset(static_folder + 'HEF_static_raw.nc')

static_raw_crop =  static_raw.where(static_raw.MASK==1, drop=True)
static_raw_crop.MASK.plot()
plt.show()
plt.close()

ds_crop = crop_file_to_glacier(static_raw)

ds_crop.MASK.plot()
plt.show()
plt.close()


def bbox_2d_array(mask, arr, varname):
    if arr.ndim == 1:
        if varname in ['time', 'Time']:
            i_min = 0
            i_max = None
        elif varname in ['lat', 'latitude']:
            ix = np.where(np.any(mask == 1, axis=1))[0]
            i_min, i_max = ix[[0, -1]]
            i_min = i_min
            i_max = i_max
        elif varname in ['lon', 'longitude']:
            ix = np.where(np.any(mask == 1, axis=0))[0]
            i_min, i_max = ix[[0, -1]]
            i_min = i_min
            i_max = i_max
            bbox = arr[i_min:i_max]
    elif arr.ndim == 2:
        ix_c = np.where(np.any(mask == 1, axis=0))[0]
        ix_r = np.where(np.any(mask == 1, axis=1))[0]
        c_min, c_max = ix_c[[0, -1]]
        r_min, r_max = ix_r[[0, -1]]

        # Draw box with one non-value border
        # Now we got bounding box -> just add +1 / +2 at every index and voila
        bbox = arr[r_min:r_max+1, c_min:c_max+1]
    elif arr.ndim == 3:
        ix_c = np.where(np.any(mask == 1, axis=0))[0]
        ix_r = np.where(np.any(mask == 1, axis=1))[0]
        c_min, c_max = ix_c[[0, -1]]
        r_min, r_max = ix_r[[0, -1]]
        bbox = arr[:, r_min:r_max, c_min:c_max]
    return bbox


dso = static_raw.copy()

print('Create cropped file.')
dso_mod = xr.Dataset()
for var in list(dso.variables):
    print(var)

    mask = dso.MASK.values
    arr2 = dso[var].values
    varname = var

    arr = bbox_2d_array(dso.MASK.values, dso[var].values, var)
    #without any indices, we are one too short on each side.. now need to figure out which side it is - create smart comparison with original dataset

    print(np.nanmin(ds_crop.lat), np.nanmax(ds_crop.lat))
    print(np.nanmin(static_raw_crop.lat), np.nanmax(static_raw_crop.lat))

    print(np.nanmin(ds_crop.lon), np.nanmax(ds_crop.lon))
    print(np.nanmin(static_raw_crop.lon), np.nanmax(static_raw_crop.lon))