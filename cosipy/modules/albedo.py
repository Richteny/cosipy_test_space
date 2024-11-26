import numpy as np

from cosipy.constants import Constants

# only required for njitted functions
albedo_method = Constants.albedo_method
albedo_mod_snow_aging = Constants.albedo_mod_snow_aging
snow_ice_threshold = Constants.snow_ice_threshold
albedo_firn = Constants.albedo_firn
albedo_fresh_snow = Constants.albedo_fresh_snow
albedo_ice = Constants.albedo_ice 
albedo_mod_snow_depth = Constants.albedo_mod_snow_depth
dt = Constants.dt
zero_temperature = Constants.zero_temperature
t_star_cutoff = Constants.t_star_cutoff
t_star_dry = Constants.t_star_dry
t_star_wet = Constants.t_star_wet
t_star_K = Constants.t_star_K

#print("You are in the albedo scheme.")
# Need to load in dic values here already to be able to call the following functions?
#opt_dict = read_opt(opt_dict, globals())

def updateAlbedo(GRID, surface_temperature, albedo_snow, opt_dict=None):
    """Updates the albedo."""
    #read_opt(opt_dict, globals())
    print("Read opt dict.")
    print(opt_dict)
    if opt_dict is not None:
        #mult_factor_RRR = opt_dict[0]
        albedo_ice = opt_dict[1]
        albedo_fresh_snow = opt_dict[2]
        albedo_firn = opt_dict[3]
        albedo_mod_snow_aging = opt_dict[4]
        albedo_mod_snow_depth = opt_dict[5]
        #center_snow_transfer_function = opt_dict[6]
        #spread_snow_transfer_function = opt_dict[7]
        #roughness_fresh_snow = opt_dict[8]
        #roughness_ice = opt_dict[9]
        #roughness_firn = opt_dict[10]
        #aging_factor_roughness = opt_dict[11]
    
    albedo_allowed = ["Oerlemans98", "Bougamont05"]
    if albedo_method == "Oerlemans98":
        alphaMod = method_Oerlemans(GRID, albedo_ice, albedo_fresh_snow, albedo_firn,
                                    albedo_mod_snow_aging, albedo_mod_snow_depth)
    elif albedo_method == "Bougamont05":
        alphaMod, albedo_snow = method_Bougamont(GRID, surface_temperature, albedo_snow)
    else:
        error_message = (
            f'Albedo method = "{albedo_method}"',
            f"is not allowed, must be one of",
            f'{", ".join(albedo_allowed)}',
        )
        raise ValueError(" ".join(error_message))

    return alphaMod, albedo_snow


def method_Oerlemans(GRID, albedo_ice: float, albedo_fresh_snow: float, albedo_firn: float, 
                     albedo_mod_snow_aging: float, albedo_mod_snow_depth: float):

    #if opt_dict is not None:
    #    mult_factor_RRR = opt_dict[0]
    #    albedo_ice = opt_dict[1]
    #    albedo_fresh_snow = opt_dict[2]
    #    albedo_firn = opt_dict[3]
    #    albedo_mod_snow_aging = opt_dict[4]
    #    albedo_mod_snow_depth = opt_dict[5]
    #    center_snow_transfer_function = opt_dict[6]
    #    spread_snow_transfer_function = opt_dict[7]
    #    roughness_fresh_snow = opt_dict[8]
    #    roughness_ice = opt_dict[9]
    #    roughness_firn = opt_dict[10]
    #    aging_factor_roughness = opt_dict[11]

    print("We have come to this point")
    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    fresh_snow_height, fresh_snow_timestamp, _ = GRID.get_fresh_snow_props()

    # Get time difference between last snowfall and now
    hours_since_snowfall = (fresh_snow_timestamp) / 3600.0

    # If fresh snow disappears faster than the snow ageing scale then set the hours_since_snowfall
    # to the old values of the underlying snowpack
    if (hours_since_snowfall < (albedo_mod_snow_aging * 24)) & (
        fresh_snow_height <= 0.0
    ):
        GRID.set_fresh_snow_props_to_old_props()
        fresh_snow_height, fresh_snow_timestamp, _ = GRID.get_fresh_snow_props()

        # Update time difference between last snowfall and now
        hours_since_snowfall = (fresh_snow_timestamp) / 3600.0

    # Check if snow or ice
    if GRID.get_node_density(0) <= snow_ice_threshold:
        # Get current snowheight from layer height
        h = GRID.get_total_snowheight()  # np.sum(GRID.get_height()[0:idx])

        # Surface albedo according to Oerlemans & Knap 1998, JGR)
        alphaSnow = albedo_firn + (
            albedo_fresh_snow - albedo_firn
        ) * np.exp((-hours_since_snowfall) / (albedo_mod_snow_aging * 24.0))
        alphaMod = alphaSnow + (albedo_ice - alphaSnow) * np.exp(
            (-1.0 * h) / (albedo_mod_snow_depth / 100.0)
        )

    else:
        # If no snow cover than set albedo to ice albedo
        alphaMod = albedo_ice

    return alphaMod


def method_Bougamont(GRID, surface_temperature, albedo_snow):
    #if opt_dict is not None:
    #    mult_factor_RRR = opt_dict[0]
    #    albedo_ice = opt_dict[1]
    #    albedo_fresh_snow = opt_dict[2]
    #    albedo_firn = opt_dict[3]
    #    albedo_mod_snow_aging = opt_dict[4]
    #    albedo_mod_snow_depth = opt_dict[5]
    #    center_snow_transfer_function = opt_dict[6]
    #    spread_snow_transfer_function = opt_dict[7]
    #    roughness_fresh_snow = opt_dict[8]
    #    roughness_ice = opt_dict[9]
    #    roughness_firn = opt_dict[10]
    #    aging_factor_roughness = opt_dict[11]

    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    _, fresh_snow_timestamp, _ = GRID.get_fresh_snow_props()

    # Get time difference between last snowfall and now:
    hours_since_snowfall = (fresh_snow_timestamp) / 3600.0

    # Convert integration time from seconds to days:
    dt_days = dt / 86400.0

    # Note: accounting for disapearance of uppermost fresh snow layer difficult due to non-constant decay rate. Unsure how to implement.

    # Get current snowheight from layer height:
    h = GRID.get_total_snowheight()

    # Check if snow or ice:
    if GRID.get_node_density(0) <= snow_ice_threshold:
        if surface_temperature >= zero_temperature:
            # Snow albedo decay timescale (t*) on a melting snow surface:
            t_star = t_star_wet
        else:
            # Snow albedo decay timescale (t*) on a dry snow surface:
            if surface_temperature < t_star_cutoff:
                t_star = (
                    t_star_dry
                    + (zero_temperature - t_star_cutoff)
                    * t_star_K
                )
            else:
                t_star = (
                    t_star_dry
                    + (zero_temperature - surface_temperature)
                    * t_star_K
                )

        # Effect of snow albedo decay due to the temporal metamorphosis of snow (Bougamont et al. 2005 - based off Oerlemans & Knap 1998):
        # Exponential function discretised in order to account for variable surface temperature-dependant decay timescales.
        albedo_snow = (
            albedo_snow - (albedo_snow - albedo_firn) / t_star * dt_days
        )

        # Reset if snowfall in current timestep
        if hours_since_snowfall == 0:
            albedo_snow = albedo_fresh_snow

        # Effect of surface albedo decay due to the snow depth (Oerlemans & Knap 1998):
        alphaMod = albedo_snow + (albedo_ice - albedo_snow) * np.exp(
            (-1.0 * h) / (albedo_mod_snow_depth / 100.0)
        )

    else:
        # If no snow cover than set albedo to ice albedo
        alphaMod = albedo_ice

    # Ensure output value is of the float data type.
    alphaMod = float(alphaMod)

    return alphaMod, albedo_snow


### idea; albedo decay like (Brock et al. 2000)? or?
### Schmidt et al 2017 >doi:10.5194/tc-2017-67, 2017 use the same albedo parameterisation from Oerlemans and Knap 1998 with a slight updated implementation of considering the surface temperature?
