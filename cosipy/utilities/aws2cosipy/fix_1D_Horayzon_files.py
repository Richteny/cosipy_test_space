import xarray as xr

# load all files by hand and re-safe # .. very crude and simple script, should be implemented in the createHORAYZON ..
ds_base = xr.open_dataset("/data/scratch/richteny/thesis/cosipy_test_space/data/static/Halji/Halji_HORAYZON-LUT_1D20m.nc")
ds_time1 = xr.open_dataset("/data/scratch/richteny/thesis/cosipy_test_space/data/static/Halji/Halji2014_HORAYZON-LUT_1D20m.nc")
ds_time2 = xr.open_dataset("/data/scratch/richteny/thesis/cosipy_test_space/data/static/Halji/Halji2018_HORAYZON-LUT_1D20m.nc")
ds_time3 = xr.open_dataset("/data/scratch/richteny/thesis/cosipy_test_space/data/static/Halji/Halji2021_HORAYZON-LUT_1D20m.nc")

ds_time1['sw_dir_cor'] = ds_time1['sw_dir_cor'].fillna(ds_base['sw_dir_cor'])
ds_time2['sw_dir_cor'] = ds_time2['sw_dir_cor'].fillna(ds_time1['sw_dir_cor'])
ds_time3['sw_dir_cor'] = ds_time3['sw_dir_cor'].fillna(ds_time2['sw_dir_cor'])

ds_time1.to_netcdf("/data/scratch/richteny/thesis/cosipy_test_space/data/static/Halji/aHalji2014_HORAYZON-LUT_1D20m.nc")
ds_time2.to_netcdf("/data/scratch/richteny/thesis/cosipy_test_space/data/static/Halji/aHalji2018_HORAYZON-LUT_1D20m.nc")
ds_time3.to_netcdf("/data/scratch/richteny/thesis/cosipy_test_space/data/static/Halji/aHalji2021_HORAYZON-LUT_1D20m.nc")

