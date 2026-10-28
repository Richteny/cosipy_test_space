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
# Snowfall mass [kg m-2] that refreshes the snow albedo by a factor 1 - 1/e (Bougamont05). Same form and default as the
# prognostic snow albedo of FSM (Essery 2015, GMD 8, 3867, Eq. 10: refresh term (Sf / S_alpha)(alpha_max - alpha),
# S_alpha = 10 kg m-2), integrated exactly over the time step.
# albedo_refresh_swe <= 0 (or missing in constants.toml) = standard COSIPY: full reset to the fresh-snow albedo at
# every snowfall event. Also sets the refresh of the fresh layer when albedo_memory = true (see below).
albedo_refresh_swe = float(getattr(Constants, "albedo_refresh_swe", 0.0))
# Albedo memory of the surface below a fresh snow layer (Bougamont05; albedo_memory = true in constants.toml).
# Closes the TODO in method_Bougamont ("account for disappearance of uppermost fresh snow layer"). Oerlemans & Knap
# (1998) blend thin snow with the surface below by exp(-d / d*) (d* = albedo_mod_snow_depth); here the same term is
# applied to a fresh layer on top of an existing snowpack:
#   - a snowfall event (>= minimum_snowfall) on an existing snowpack stores the snow albedo of that surface (the
#     'albedo of the underlying surface before the snowfall', Kayastha et al. 1999, J. Glaciol. 45(151), Eq. 3); the
#     fresh layer is reset to the fresh-snow albedo (as in Oerlemans & Knap 1998 and the original Bougamont05 branch)
#     and then ages with Bougamont's t*; the stored surface does not age (it is buried)
#   - surface albedo = a_fresh_layer + (a_below - a_fresh_layer) * exp(-h_fresh / d*), where a_below is the stored
#     snow albedo blended with ice by the depth of the old snow (the existing Oerlemans & Knap term)
#   - melt and sublimation thin the fresh layer (tracked in cosipy_core); when it is gone, the stored albedo returns
#   - at h_fresh > 3 d* the surface below contributes < 5 % (Oerlemans & Knap 1998) -> the fresh layer becomes the
#     snowpack and the stored albedo is dropped
# With albedo_refresh_swe > 0 (memory + FSM refresh), a snowfall refreshes the current snow albedo by the share
# 1 - exp(-SWE / albedo_refresh_swe) instead of the full reset. That share already contains the contribution of the
# surface below, so the exp(-h_fresh / d*) blend is not applied on top (it would count thin snow twice); the
# memory is the stored albedo that returns when the fresh snow is gone. Merge at fresh SWE > 3 albedo_refresh_swe
# (surface below < 5 %, same convention as 3 d*).
# No new parameter. With albedo_memory = false the results are identical to before.
albedo_memory = bool(getattr(Constants, "albedo_memory", False))
MEMORY_MERGE_DEPTH_FACTOR = 3.0
# Superimposed ice (superimposed_ice = true, see refreezing.py): while superimposed ice formed by refreezing lies at
# the surface, bare ice and the surface below thin snow have albedo_superimposed_ice instead of albedo_ice
# (fixed value, not calibrated; albedo_ice itself is unchanged). No model precedent was found for a separate
# superimposed-ice albedo; the value is set from Landsat (Dongkemadi tongue after refreezing events, ~0.45-0.6).
albedo_superimposed_ice = float(getattr(Constants, "albedo_superimposed_ice", 0.6))
# Snow albedo on bare ice (Bougamont05; bug fix 27 Oct 2026, always on). The snow albedo is a state carried from step
# to step. While the surface is bare ice it used to keep the value the last snow had when it melted away (often aged,
# ~0.5-0.6). With the mass-weighted refresh (albedo_refresh_swe > 0) a new snowfall on the ice then only refreshed that
# stale value by 1 - exp(-SWE / albedo_refresh_swe) (2 kg m-2 -> 18 %): new snow on ice started as old snow, and the
# surface could not get brighter than the stale value, whatever albedo_mod_snow_depth was. The refresh describes new
# snow mixing optically with an existing snowpack; on bare ice there is none. Now the snow albedo is set to
# albedo_fresh_snow while the surface is bare ice, so new snow on ice starts fresh and exp(-h / d*) alone decides how
# much of the ice shows through. Original COSIPY (full reset at every snowfall, albedo_refresh_swe = 0) never used the
# stale value, so its results are unchanged.


def updateAlbedo(
    GRID, surface_temperature: float, albedo_snow: float, opt_dict=None, fresh_swe: float = 0.0,
    memory_state=None, superimposed: bool = False) -> tuple:
    """Update the surface albedo.

    Implemented albedo methods:

        - **Oerlemans98**: Oerlemans & Knap (1998)
        - **Bougamont05**: Bougamont et al. (2005)

    Args:
        GRID (Grid): Glacier data structure.
        surface_temperature: Surface temperature.
        albedo_snow: Initial snow albedo.
        opt_dict (tuple): Tuple of parameters.
        fresh_swe: Snowfall of this time step [kg m-2] (Bougamont05 albedo refresh).
        memory_state (list): [stored snow albedo below the fresh layer (nan = none), fresh-layer SWE [kg m-2],
            fresh-layer height [m]]; updated in place (albedo memory, Bougamont05 only; None = off).
        superimposed: superimposed ice at the top of the ice -> albedo_superimposed_ice replaces the ice albedo.

    Returns:
        tuple[float,float]: Updated surface albedo and snow albedo.

    Raises:
        NotImplementedError: Albedo method is not allowed.
    """
    if opt_dict is not None:
        albedo_ice = opt_dict[1]
        albedo_fresh_snow = opt_dict[2]
        albedo_firn = opt_dict[3]
        albedo_mod_snow_aging = opt_dict[4]
        albedo_mod_snow_depth = opt_dict[5]
        t_star_wet = opt_dict[15]
        t_star_dry = opt_dict[16]
        t_star_K = opt_dict[17]
    else:
        albedo_ice = Constants.albedo_ice
        albedo_fresh_snow = Constants.albedo_fresh_snow
        albedo_firn = Constants.albedo_firn
        albedo_mod_snow_aging = Constants.albedo_mod_snow_aging
        albedo_mod_snow_depth = Constants.albedo_mod_snow_depth
        t_star_dry = Constants.t_star_dry
        t_star_wet = Constants.t_star_wet
        t_star_K = Constants.t_star_K

    albedo_below = None                                # None: the module-level albedo_ice (as before)
    if superimposed:
        albedo_ice = albedo_superimposed_ice
        albedo_below = albedo_superimposed_ice

    albedo_allowed = ["Oerlemans98", "Bougamont05"]
    if albedo_method == "Oerlemans98":
        alphaMod = method_Oerlemans(GRID, albedo_ice, albedo_fresh_snow, albedo_firn,
                                    albedo_mod_snow_aging, albedo_mod_snow_depth, albedo_below)
    elif albedo_method == "Bougamont05":
        alphaMod, albedo_snow = method_Bougamont(GRID, surface_temperature, albedo_snow, albedo_ice, albedo_fresh_snow,
                                                 albedo_firn, albedo_mod_snow_depth, t_star_wet, t_star_dry, t_star_K,
                                                 fresh_swe, memory_state, albedo_below)
    else:
        error_message = (
            f'Albedo method = "{albedo_method}"',
            f"is not allowed, must be one of",
            f'{", ".join(albedo_allowed)}',
        )
        raise ValueError(" ".join(error_message))

    return alphaMod, albedo_snow


def get_surface_properties(GRID) -> tuple:
    """Get snowpack properties.

    Args:
        GRID (Grid): Glacier data structure.

    Returns:
        tuple[float, float, float]: Height and timestamp of fresh snow,
        and the hours elapsed since the last snowfall.
    """

    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    fresh_snow_height, fresh_snow_timestamp, _ = GRID.get_fresh_snow_props()

    # Get time difference between last snowfall and now
    hours_since_snowfall = fresh_snow_timestamp / 3600.0

    """If fresh snow disappears faster than the snow ageing scale then
    set the hours_since_snowfall to the old values of the underlying
    snowpack."""
    if (hours_since_snowfall < (albedo_mod_snow_aging * 24)) & (
        fresh_snow_height <= 0.0
    ):
        GRID.set_fresh_snow_props_to_old_props()
        (
            fresh_snow_height,
            fresh_snow_timestamp,
            _,
        ) = GRID.get_fresh_snow_props()

        # Update time difference between last snowfall and now
        hours_since_snowfall = fresh_snow_timestamp / 3600.0

    return fresh_snow_height, fresh_snow_timestamp, hours_since_snowfall


def get_surface_properties(GRID) -> tuple:
    """Get snowpack properties.

    Args:
        GRID (Grid): Glacier data structure.

    Returns:
        tuple[float, float, float]: Height and timestamp of fresh snow,
        and the hours elapsed since the last snowfall.
    """

    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    fresh_snow_height, fresh_snow_timestamp, _ = GRID.get_fresh_snow_props()

    # Get time difference between last snowfall and now
    hours_since_snowfall = fresh_snow_timestamp / 3600.0

    # If fresh snow disappears faster than the snow ageing scale then
    # set the hours_since_snowfall to the old values of the underlying
    # snowpack
    if (hours_since_snowfall < (albedo_mod_snow_aging * 24)) & (
        fresh_snow_height <= 0.0
    ):
        GRID.set_fresh_snow_props_to_old_props()
        (
            fresh_snow_height,
            fresh_snow_timestamp,
            _,
        ) = GRID.get_fresh_snow_props()

        # Update time difference between last snowfall and now
        hours_since_snowfall = fresh_snow_timestamp / 3600.0

    return fresh_snow_height, fresh_snow_timestamp, hours_since_snowfall


def get_simple_albedo(elapsed_time: float, albedo_firn: float, albedo_fresh_snow: float,
    albedo_mod_snow_aging: float) -> float:
    """Get surface albedo neglecting snowpack depth.

    From Oerlemans & Knap (1998).

    Args:
        elapsed_time: Hours elapsed since last snowfall.

    Returns:
        Surface albedo without accounting for snowpack depth.
    """
    albedo = albedo_firn + (albedo_fresh_snow - albedo_firn) * np.exp(
        (-elapsed_time) / (albedo_mod_snow_aging * 24.0)
    )

    return albedo


def get_albedo_with_decay(snow_albedo: float, snow_height: float, albedo_mod_snow_depth: float,
                          albedo_below=None) -> float:
    """Apply surface albedo decay due to the snow depth.

    Taken from Oerlemans & Knap (1998).

    Args:
        snow_alb: Initial snow albedo.
        snow_height: Height of snowpack.
        albedo_below: albedo of the surface below the snow (None = albedo_ice).

    Returns:
        Surface albedo with snow depth decay.
    """
    a_below = albedo_ice if albedo_below is None else albedo_below
    albedo = snow_albedo + (a_below - snow_albedo) * np.exp(
        (-1.0 * snow_height) / (albedo_mod_snow_depth / 100.0)
    )

    return albedo


def method_Oerlemans(GRID, albedo_ice: float, albedo_fresh_snow: float, albedo_firn: float, 
                     albedo_mod_snow_aging: float, albedo_mod_snow_depth: float, albedo_below=None):
    """Get surface albedo using method from Oerlemans & Knap (1998).

    Args:
        GRID (Grid): Glacier data structure.
        albedo_ice: ice albedo parameter.
        albedo_fresh_snow: parameter for fresh snow albedo.
        albedo_firn: parameter for firn albedo.
        albedo_mod_snow_aging: parameter for the albedo aging scale.
        albedo_mod_snow_depth: parameter for the albedo depth scale.

    Returns:
        Surface albedo.
    """
    _, _, hours_since_snowfall = get_surface_properties(GRID)

    
    #Parse params locally
    #albedo_ice = albedo_ice
    #albedo_fresh_snow = albedo_fresh_snow
    #albedo_firn = albedo_firn
    #albedo_mod_snow_aging = albedo_mod_snow_aging
    #albedo_mod_snow_depth = albedo_mod_snow_depth
    # Check if snow or ice
    if GRID.get_node_density(0) <= snow_ice_threshold:
        # Get current snowheight from layer height
        h = GRID.get_total_snowheight()  # np.sum(GRID.get_height()[0:idx])
        
        # Surface albedo according to Oerlemans & Knap 1998, JGR)
        alphaSnow = get_simple_albedo(elapsed_time=hours_since_snowfall, albedo_firn=albedo_firn,
                                      albedo_fresh_snow=albedo_fresh_snow, albedo_mod_snow_aging=albedo_mod_snow_aging)
        alphaMod = get_albedo_with_decay(snow_albedo=alphaSnow, snow_height=h, albedo_mod_snow_depth=albedo_mod_snow_depth,
                                         albedo_below=albedo_below)

    else:
        # If no snow cover than set albedo to ice albedo
        alphaMod = albedo_ice

    return alphaMod


def method_Bougamont(GRID, surface_temperature, albedo_snow, albedo_ice: float, albedo_fresh_snow: float, 
                     albedo_firn: float, albedo_mod_snow_depth: float, t_star_wet: float, t_star_dry: float, t_star_K: float,
                     fresh_swe: float = 0.0, memory_state=None, albedo_below=None):
    """Get surface and snow albedos using method from Bougamont (2005).

    Args:
        GRID (Grid): Glacier data structure.
        surface_temperature: Surface temperature.
        albedo_snow: Initial snow albedo.
        albedo_ice: ice albedo parameter.
        albedo_fresh_snow: parameter for fresh snow albedo.
        albedo_firn: parameter for firn albedo.
        albedo_mod_snow_depth: parameter for the depth scale.
        fresh_swe: Snowfall of this time step [kg m-2].
        memory_state (list): [stored snow albedo below the fresh layer (nan = none), fresh-layer SWE [kg m-2],
            fresh-layer height [m]], updated in place; None = no memory (see albedo_memory at the top of this module).

    Returns:
        tuple[float, float]: Updated surface and snow albedos. With memory, the snow albedo is the albedo of the
        fresh layer while one exists, else of the snowpack.

    TODO (original): Account for disappearance of uppermost fresh snow layer is
    difficult due to non-constant decay rate. -> addressed by the optional albedo memory (albedo_memory).
    """
    use_memory = memory_state is not None
    # Get hours since the last snowfall
    # First get fresh snow properties (height and timestamp)
    _, _, hours_since_snowfall = get_surface_properties(GRID)

    # Convert integration time from seconds to days:
    dt_days = dt / 86400.0

    # Check if snow or ice:
    if GRID.get_node_density(0) <= snow_ice_threshold:
        # Get current snowheight from layer height
        h = GRID.get_total_snowheight()

        if surface_temperature >= zero_temperature:
            # Snow albedo decay timescale (t*) on a melting snow surface:
            t_star = t_star_wet
        else:
            # Snow albedo decay timescale (t*) on a dry snow surface:
            if surface_temperature < t_star_cutoff:
                t_star = (
                    t_star_dry + (zero_temperature - t_star_cutoff) * t_star_K
                )
            else:
                t_star = (
                    t_star_dry
                    + (zero_temperature - surface_temperature) * t_star_K
                )

        """Effect of snow albedo decay due to the temporal metamorphosis of
        snow (Bougamont et al. 2005 - based off Oerlemans & Knap 1998):
        Exponential function discretised to account for variable surface
        temperature-dependant decay timescales."""

        # slightly faster than one-liner
        delta_albedo = albedo_snow - albedo_firn
        albedo_snow -= delta_albedo * float(dt_days) / float(t_star)

        if use_memory:
            albedo_under, swe_fresh, h_fresh = memory_state[0], memory_state[1], memory_state[2]
            d_star = albedo_mod_snow_depth / 100.0                     # [m]
            if not np.isnan(albedo_under) and swe_fresh <= 0.0:
                # fresh layer melted / sublimated away: the stored surface returns
                albedo_snow, albedo_under, swe_fresh, h_fresh = albedo_under, np.nan, 0.0, 0.0
            if hours_since_snowfall == 0 and fresh_swe > 0.0:
                d_new, _, _ = GRID.get_fresh_snow_props()          # this step's fresh snow height [m]
                if np.isnan(albedo_under) and (h - d_new) > 0.0:
                    albedo_under = albedo_snow                     # new event on an existing snowpack: store it
                    swe_fresh, h_fresh = 0.0, 0.0
                if albedo_refresh_swe > 0.0:
                    # mass-weighted refresh (FSM) of the current snow albedo
                    w = 1.0 - np.exp(-fresh_swe / albedo_refresh_swe)
                    albedo_snow = albedo_snow + w * (albedo_fresh_snow - albedo_snow)
                else:
                    albedo_snow = albedo_fresh_snow                # fresh layer reset (Oerlemans & Knap 1998)
                if not np.isnan(albedo_under):
                    swe_fresh, h_fresh = swe_fresh + fresh_swe, h_fresh + d_new
            if albedo_refresh_swe > 0.0:
                # refresh share 1 - exp(-SWE/S) > 95 % at SWE > 3 S: the surface below no longer matters
                merged = swe_fresh > MEMORY_MERGE_DEPTH_FACTOR * albedo_refresh_swe
            else:
                # optically thick (< 5 % from below, Oerlemans & Knap 1998)
                merged = h_fresh > MEMORY_MERGE_DEPTH_FACTOR * d_star
            if not np.isnan(albedo_under) and merged:
                albedo_under, swe_fresh, h_fresh = np.nan, 0.0, 0.0   # the fresh layer becomes the snowpack
            memory_state[0], memory_state[1], memory_state[2] = albedo_under, swe_fresh, h_fresh
            if not np.isnan(albedo_under) and albedo_refresh_swe <= 0.0:
                h_old = max(h - h_fresh, 0.0)
                a_below = albedo_under + (albedo_ice - albedo_under) * np.exp(-h_old / d_star)
                alphaMod = albedo_snow + (a_below - albedo_snow) * np.exp(-h_fresh / d_star)
            else:
                alphaMod = get_albedo_with_decay(snow_albedo=albedo_snow, snow_height=h,
                                                 albedo_mod_snow_depth=albedo_mod_snow_depth,
                                                 albedo_below=albedo_below)
            return float(alphaMod), albedo_snow

        # Refresh by this step's snowfall:
        #  albedo_refresh_swe > 0: share w = 1 - exp(-SWE / albedo_refresh_swe) of the gap to fresh-snow albedo is
        #                          closed (FSM, Essery 2015, Eq. 10, integrated exactly over the step)
        #  albedo_refresh_swe = 0: standard COSIPY -- full reset to fresh-snow albedo at every snowfall event
        if hours_since_snowfall == 0:
            if albedo_refresh_swe > 0.0:
                w = 1.0 - np.exp(-fresh_swe / albedo_refresh_swe) if fresh_swe > 0.0 else 0.0
                albedo_snow = albedo_snow + w * (albedo_fresh_snow - albedo_snow)
            else:
                albedo_snow = albedo_fresh_snow

        # Effect of surface albedo decay due to the snow depth (Oerlemans & Knap 1998):
        alphaMod = get_albedo_with_decay(snow_albedo=albedo_snow,snow_height=h,albedo_mod_snow_depth=albedo_mod_snow_depth,
                                         albedo_below=albedo_below)

    else:
        # If no snow cover than set albedo to ice albedo
        alphaMod = albedo_ice
        if use_memory:                                    # bare ice: nothing stored
            memory_state[0], memory_state[1], memory_state[2] = np.nan, 0.0, 0.0
        albedo_snow = albedo_fresh_snow                   # no snow left: the next snowfall starts as fresh snow

    # Ensure output value is of the float data type.
    alphaMod = float(alphaMod)  # faster than isinstance

    return alphaMod, albedo_snow


### idea; albedo decay like (Brock et al. 2000)? or?
### Schmidt et al 2017 >doi:10.5194/tc-2017-67, 2017 use the same albedo parameterisation from Oerlemans and Knap 1998 with a slight updated implementation of considering the surface temperature?
