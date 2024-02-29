from config import *
from constants import *
import xarray as xr
from COSIPY import prereq_res
# Load sample file
result = xr.open_dataset()

tsla_observations = pd.read_csv(tsl_data_file)
dates, clean_day_vals, secs, holder = prereq_res(result.sel(time=slice("2000-01-01","2009-12-31")))
resampled_array = resample_by_hand(holder, result.SNOWHEIGHT.values, secs, clean_day_vals)
resampled_out = construct_resampled_ds(result, resampled_array, dates.values)
resampled_out['HGT'] = (('lat','lon'), result['HGT'].data)
resampled_out['MASK'] = (('lat','lon'), result['MASK'].data)

tsl_out = create_tsl_df(resampled_out, min_snowheight, tsl_method, tsl_normalize)


## other version 
a_resampled_out = resample_output(result)
a_tsl_out = create_tsl_df(a_resampled_out, min_snowheight, tsl_method, tsl_normalize)

