from cosipy.constants import Constants
from cosipy.config import Config
from numba import njit, typeof
from numba.core import types
from numba.typed import Dict
# to support numba, create a typed dictionary consisting of only one type of value
# Dict to hold valid options.

albfrsnow = Constants.albedo_fresh_snow
albfirn = Constants.albedo_firn
albice = Constants.albedo_ice
rfrs = Constants.roughness_fresh_snow
rfirn = Constants.roughness_firn
rice = Constants.roughness_ice
RRRfactor = Constants.mult_factor_RRR
albmodsnowage =  Constants.albedo_mod_snow_aging
albmodsnowdep = Constants.albedo_mod_snow_depth
cstf = Constants.center_snow_transfer_function
sstf = Constants.spread_snow_transfer_function

OPTIONS = Dict.empty(key_type=types.unicode_type,
                     value_type=types.float64,
                     )
OPTIONS['albedo_fresh_snow'] = albfrsnow
OPTIONS['albedo_firn'] = albfirn
OPTIONS['albedo_ice'] = albice
OPTIONS['roughness_fresh_snow'] = rfrs
OPTIONS['roughness_ice'] = rice
OPTIONS['roughness_firn'] = rfirn
OPTIONS['mult_factor_RRR'] = RRRfactor
OPTIONS['albedo_mod_snow_aging'] = albmodsnowage
OPTIONS['albedo_mod_snow_depth'] =  albmodsnowdep
OPTIONS['center_snow_transfer_function'] = cstf
OPTIONS['spread_snow_transfer_function'] = sstf

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


#@njit
#def read_opt(opt_dict, glob):
#    """ Reads the opt_dict and overwrites the key-value pairs in glob - the calling function's
#    globals() dictionary."""
    #print("Called readopt func. Type of dictionary is:")
    #print(typeof(opt_dict))
#    if opt_dict is not None:
#        #print(opt_dict)
#        for key in opt_dict:
#            if key in OPTIONS.keys(): 
#                glob[key] = opt_dict[key]
#            else:
#                print(f'ATTENTION: {key} is not a valid option. Default will be used!')

def read_opt(opt_dict, glob):
    print("testing. this is read_opt")
