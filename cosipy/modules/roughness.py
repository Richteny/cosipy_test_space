from cosipy.constants import Constants


def updateRoughness(GRID, opt_dict=None) -> float:
    """Update the surface roughness length.

    Implemented methods:

        - **Moelg12**: Linear increase in snow roughness length over
          time. From Mölg et al. (2009).

    Args:
        GRID (Grid): Glacier data structure.

    Returns:
        Updated surface roughness length [mm].
    """
    if opt_dict is not None:
        roughness_fresh_snow = opt_dict[8]
        roughness_ice = opt_dict[9]
        roughness_firn = opt_dict[10]
        aging_factor_roughness = opt_dict[11]
    else:
        roughness_fresh_snow = Constants.roughness_fresh_snow
        roughness_ice = Constants.roughness_ice
        roughness_firn = Constants.roughness_firn
        aging_factor_roughness = Constants.aging_factor_roughness

    roughness_allowed = ["Moelg12"]
    if Constants.roughness_method == "Moelg12":
        sigma = method_Moelg(GRID, roughness_fresh_snow, aging_factor_roughness, roughness_firn, roughness_ice)
    else:
        error_message = (
            f'Roughness method = "{Constants.roughness_method}" is not allowed,',
            f'must be one of {", ".join(roughness_allowed)}',
        )
        raise ValueError(" ".join(error_message))

    return sigma


def method_Moelg(GRID, roughness_fresh_snow: float, aging_factor_roughness: float,
    roughness_firn: float, roughness_ice: float) -> float:
    """Update the roughness length.

    Adapted from Moelg et al. (2009), J.Clim. The roughness length of
    snow linearly increases from 0.24 (fresh snow) to 4 (firn) in
    60 days (1440 hours) i.e. (4-0.24)/1440 = 0.0026.

    Args:
        GRID (Grid): Glacier data structure.

    Returns:
        Surface roughness length, [mm]
    """
    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    _, fresh_snow_timestamp, _ = GRID.get_fresh_snow_props()

    # Get time difference between last snowfall and now
    hours_since_snowfall = fresh_snow_timestamp / 3600.0

    # Check whether snow or ice
    if GRID.get_node_density(0) <= Constants.snow_ice_threshold:
        sigma = min(
            roughness_fresh_snow
            + aging_factor_roughness * hours_since_snowfall,
            roughness_firn,
        )
    else:
        sigma = roughness_ice  # Roughness length, set to ice

    return sigma / 1000
