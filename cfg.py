"""
This is the COSIPY configuration file. It reads in constants
and variables from constants.cfg and config.cfg and make
their contents available as dictionaries.
"""
from configobj import ConfigObj
from numba import types
from numba.typed import Dict


def init_config():
    # Read in the config.cfg file as an config object.
    cp = ConfigObj('./config.cfg')

    # Deep copy to a dict. strings are in the correct format, have to cast
    # others.
    config = cp.dict()

    # Set bools
    config['restart'] = cp.as_bool('restart')
    config['stake_evaluation'] = cp.as_bool('stake_evaluation')
    config['WRF'] = cp.as_bool('WRF')
    config['WRF_X_CSPY'] = cp.as_bool('WRF_X_CSPY')
    config['slurm_use'] = cp.as_bool('slurm_use')
    config['full_field'] = cp.as_bool('full_field')
    config['force_use_TP'] = cp.as_bool('force_use_TP')
    config['force_use_N'] = cp.as_bool('force_use_N')
    config['tile'] = cp.as_bool('tile')

    # Set ints
    config['compression_level'] = cp.as_int('compression_level')
    config['local_port'] = cp.as_int('local_port')
    config['xstart'] = cp.as_int('xstart')
    config['xend'] = cp.as_int('xend')
    config['ystart'] = cp.as_int('ystart')
    config['yend'] = cp.as_int('yend')

    # We do the logic here instead of in the config. Would be nice if this was
    # updated if a user later changes WRF. Requires an ordered dict, see OGGM
    # cfg.py. Basically an class extended from ordered dicts.
    if config['workers'] == 'None':
        config['workers'] = None
    else:
        config['workers'] = cp.as_int('workers')

    # Change coordinates if WRF
    if config['WRF']:
        # name of dimension in WRF in- and output
        config['northing'] = 'south_north'
        # name of dimension in WRF in- and output
        config['easting'] = 'west_east'

    if config['WRF_X_CSPY']:
        config['full_field'] = True

    config['time_start_str'] = config['time_start'][0:10].replace('-', '')
    config['time_end_str'] = config['time_end'][0:10].replace('-', '')
    config['output_netcdf'] = 'Zhadang_ERA5_' + config['time_start_str'] +\
                              '-' + config['time_end_str'] + '.nc'

    return config


def init_constants(config):
    # Parse the constants.cfg file into a dictionary. Casts necessary variables
    # into the correct types.
    cp = ConfigObj('./constants.cfg')
    # Create a dict out of the configobj
    constants = cp.dict()
    # We try and cast all values to floats
    for key, value in constants.items():
        try:
            constants[key] = float(value)
        # If it doesn't work, we keep the original i.e. a string. This works
        # because constants.cfg only contains strings and numbers.
        except TypeError:
            pass
        except ValueError:
            pass
    # Some edge cases where we don't want float.
    constants['max_layers'] = int(constants['max_layers'])
    # Some changes if using wrf. As for config, would be nice if this changed
    # automatically if changed during runtime.
    if config['WRF_X_CSPY']:
        constants['stability_correction'] = 'MO'
        constants['sfc_temperature_method'] = 'Newton'

    return constants


def get_typed_dicts(NAMELIST):
    '''Utility function used to split the namelist into typed numba dicts.'''

    # Typed dict for all floats
    d_float = Dict.empty(key_type=types.unicode_type,
                         value_type=types.float64)
    # Typed dict for all integers.
    d_int = Dict.empty(key_type=types.unicode_type,
                       value_type=types.intp)
    # Typed dict for strings.
    d_str = Dict.empty(key_type=types.unicode_type,
                       value_type=types.unicode_type)
    # Typed dict for booleans.
    d_bool = Dict.empty(key_type=types.unicode_type,
                        value_type=types.boolean)

    # Loop over the NAMELIST and assign keys value pairs to the corresponding
    # typed dict.
    for k, v in NAMELIST.items():
        if type(v) == float:
            d_float[k] = v
        elif type(v) == int:
            d_int[k] = v
        elif type(v) == str:
            d_str[k] = v
        elif type(v) == bool:
            d_bool[k] = v

    return d_float, d_int, d_str, d_bool


def init_main_dict():
    # We merge the two dicts for simpler use throughout the model.
    config = init_config()
    constants = init_constants(config)
    NAMELIST = {**config, **constants}

    return NAMELIST


NAMELIST = init_main_dict()
