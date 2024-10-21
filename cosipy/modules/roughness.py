from cosipy.constants import Constants

# need to define variables locally to be able to change them with our dictionary
roughness_fresh_snow = Constants.roughness_fresh_snow
aging_factor_roughness = Constants.aging_factor_roughness
roughness_firn = Constants.roughness_firn
roughness_ice = Constants.roughness_ice


def updateRoughness(GRID, opt_dict=None):

    # Read and set options
    if opt_dict is not None:
        mult_factor_RRR = opt_dict[0]
        albedo_ice = opt_dict[1]
        albedo_fresh_snow = opt_dict[2]
        albedo_firn = opt_dict[3]
        albedo_mod_snow_aging = opt_dict[4]
        albedo_mod_snow_depth = opt_dict[5]
        center_snow_transfer_function = opt_dict[6]
        spread_snow_transfer_function = opt_dict[7]
        roughness_fresh_snow = opt_dict[8]
        roughness_ice = opt_dict[9]
        roughness_firn = opt_dict[10]
        aging_factor_roughness = opt_dict[11]
    #read_opt(opt_dict, globals())
    roughness_allowed = ['Moelg12']
    if Constants.roughness_method == 'Moelg12':
        sigma = method_Moelg(GRID)
    else:
        error_message = (
            f'Roughness method = "{Constants.roughness_method}" is not allowed,',
            f'must be one of {", ".join(roughness_allowed)}'
        )
        raise ValueError(" ".join(error_message))

    return sigma


def method_Moelg(GRID):
    """Update the roughness length (Moelg et al 2009, J.Clim.)."""

    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    fresh_snow_height, fresh_snow_timestamp, _  = GRID.get_fresh_snow_props()

    # Get time difference between last snowfall and now
    hours_since_snowfall = (fresh_snow_timestamp)/3600.0

    # Check whether snow or ice
    if (GRID.get_node_density(0) <= Constants.snow_ice_threshold):

        # Roughness length linear increase from 0.24 (fresh snow) to 4 (firn) in 60 days (1440 hours); (4-0.24)/1440 = 0.0026
        sigma = min(roughness_fresh_snow + aging_factor_roughness * hours_since_snowfall, roughness_firn)

    else:

        # Roughness length, set to ice
        sigma = roughness_ice

    return (sigma / 1000)
