from cosipy.constants import Constants
from cosipy.config import Config
from numba import njit, typeof
from numba.core import types
from numba.typed import Dict
# to support numba, create a typed dictionary consisting of only one type of value
# Dict to hold valid options.

OPTIONS = Dict.empty(key_type=types.unicode_type,
                     value_type=types.float64,
                     )
OPTIONS['albedo_fresh_snow'] = Constants.albedo_fresh_snow
OPTIONS['albedo_firn'] = Constants.albedo_firn
OPTIONS['albedo_ice'] = Constants.albedo_ice
OPTIONS['roughness_fresh_snow'] = Constants.roughness_fresh_snow
OPTIONS['roughness_ice'] = Constants.roughness_ice
OPTIONS['roughness_firn'] = Constants.roughness_firn
OPTIONS['mult_factor_RRR'] = Constants.mult_factor_RRR
OPTIONS['albedo_mod_snow_aging'] = Constants.albedo_mod_snow_aging
OPTIONS['albedo_mod_snow_depth'] =  Constants.albedo_mod_snow_depth
OPTIONS['center_snow_transfer_function'] = Constants.center_snow_transfer_function
OPTIONS['spread_snow_transfer_function'] = Constants.spread_snow_transfer_function

#OPTIONS = {
#    'stability_correction': Constants.stability_correction,
#    'albedo_method': Constants.albedo_method,
#    'densification_method': Constants.densification_method,
#    'penetrating_method': Constants.penetrating_method,
#    'roughness_method': Constants.roughness_method,
#    'saturation_water_vapour_method': Constants.saturation_water_vapour_method,
#    'thermal_conductivity_method': Constants.thermal_conductivity_method,
#    'sfc_temperature_method': Constants.sfc_temperature_method,
#    'albedo_fresh_snow': Constants.albedo_fresh_snow,
#    'albedo_firn': Constants.albedo_firn,
#    'albedo_ice': Constants.albedo_ice,
#    'roughness_fresh_snow': Constants.roughness_fresh_snow,
#    'roughness_ice': Constants.roughness_ice,
#    'roughness_firn': Constants.roughness_firn,
#    'time_end': Config.time_end,
#    'time_start': Config.time_start,
#    'input_netcdf': Config.input_netcdf,
#    'output_netcdf': Config.output_prefix,
#    'mult_factor_RRR': Constants.mult_factor_RRR,
#    'albedo_mod_snow_aging': Constants.albedo_mod_snow_aging,
#    'albedo_mod_snow_depth': Constants.albedo_mod_snow_depth,
#    'center_snow_transfer_function' : Constants.center_snow_transfer_function,
#    'spread_snow_transfer_function': Constants.spread_snow_transfer_function
#    }


@njit
def read_opt(opt_dict, glob):
    """ Reads the opt_dict and overwrites the key-value pairs in glob - the calling function's
    globals() dictionary."""
    #print("Called readopt func. Type of dictionary is:")
    #print(typeof(opt_dict))
    if opt_dict is not None:
        #print(opt_dict)
        for key in opt_dict:
            if key in OPTIONS.keys(): 
                glob[key] = opt_dict[key]
            else:
                print(f'ATTENTION: {key} is not a valid option. Default will be used!')
