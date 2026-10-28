from numba import njit

from cosipy.constants import Constants

zero_temperature = Constants.zero_temperature
air_density = Constants.air_density
ice_density = Constants.ice_density
water_density = Constants.water_density
spec_heat_ice = Constants.spec_heat_ice
spec_heat_water = Constants.spec_heat_water
lat_heat_melting = Constants.lat_heat_melting
snow_ice_threshold = Constants.snow_ice_threshold
pore_closeoff_density = Constants.pore_closeoff_density

# Superimposed ice (superimposed_ice = true in constants.toml; false = identical to before).
# Before: refreezing (below) is limited by pore close-off, so glacier ice takes up no water; water reaching the ice
# passes through the ice layers (irreducible water 0) and leaves the column as runoff in the same time step, however
# cold the ice. A snow layer refrozen to close-off on the ice stays 'snow' (830 kg m-3 < snow_ice_threshold).
# Superimposed ice forms exactly there (Wakahama et al. 1976; Fujita et al. 1996, Xiao Dongkemadi: ~60 % of the
# May-Sep meltwater in the accumulation zone refroze, ~26 % of the snow became superimposed ice).
# Two processes, no new parameter:
#   1. interface refreezing (percolation.py): water that has passed through (or came from) snow and reaches the first
#      ice layer refreezes onto it. Per time step the amount is the smallest of
#        - the water arriving at the interface,
#        - the cold content of the first ice layer (warmed to at most 0 deg C; Fujita & Ageta 2000: superimposed ice
#          from the change in glacier-ice temperature; van Pelt et al. 2012: refreezing until the melting point),
#        - the heat conducted into the ice over the time step, k (T0 - T) / (h / 2) dt, with the interface at 0 deg C
#          and the layer's own conductivity (the interface temperature gradient limits the rate: Wakahama et al.
#          1976, as used by Wright et al. 2005); this keeps the rate independent of the ice layer thickness.
#      The rest runs off as before (no ponding / slush storage).
#   2. superimposition: a snow layer lying directly on the ice that refreezing in this step has brought to pore
#      close-off (830 kg m-3, impermeable as in GEMB, Gardner et al. 2023) becomes ice (mass conserved, height
#      reduced to ice density); EBFM (van Pelt et al. 2012) likewise lets refreezing raise layers up to ice density.
# The mass formed is tracked in cosipy_core; while it is at the surface its albedo is albedo_superimposed_ice
# (albedo.py; the ice albedo albedo_ice is unchanged).
superimposed_ice = bool(getattr(Constants, "superimposed_ice", False))
spec_heat_ice_c = Constants.spec_heat_ice
icf_ice_layer = (snow_ice_threshold - air_density) / (ice_density - air_density)
phi_ice_closeoff = (pore_closeoff_density - air_density) / (ice_density - air_density)


@njit
def interface_refreeze_limit(GRID, idx: int, dt: float) -> float:
    """Water [m w.e.] that can refreeze onto ice layer idx in one time step: the smaller of its cold content (warmed
    to at most 0 deg C) and the heat conducted from the 0 deg C interface into the layer over dt."""
    T = GRID.get_node_temperature(idx)
    if T >= zero_temperature:
        return 0.0
    h = GRID.get_node_height(idx)
    m = GRID.get_node_ice_fraction(idx) * ice_density * h                                # ice mass [kg m-2]
    cold = spec_heat_ice_c * m * (zero_temperature - T)                                   # [J m-2]
    cond = GRID.get_node_thermal_conductivity(idx) * (zero_temperature - T) / (0.5 * h) * dt   # [J m-2]
    return min(cold, cond) / (water_density * lat_heat_melting)


@njit
def add_refrozen_to_ice(GRID, idx: int, r: float):
    """Refreeze r [m w.e.] onto ice layer idx: mass added at the layer's ice fraction (height grows), latent heat
    warms the layer (new ice formed at 0 deg C mixes with the old ice), liquid water of the layer kept in mass."""
    if r <= 0.0:
        return
    h = GRID.get_node_height(idx)
    icf = GRID.get_node_ice_fraction(idx)
    T = GRID.get_node_temperature(idx)
    w = GRID.get_node_liquid_water_content(idx) * h
    m = icf * ice_density * h
    dm = r * water_density
    h_new = h + dm / (icf * ice_density)
    T_new = (spec_heat_ice_c * m * T + dm * lat_heat_melting + spec_heat_ice_c * dm * zero_temperature) \
        / (spec_heat_ice_c * (m + dm))
    GRID.set_node_height(idx, h_new)
    GRID.set_node_liquid_water_content(idx, w / h_new)
    GRID.set_node_temperature(idx, min(T_new, zero_temperature))


@njit
def superimpose_refrozen_snow(GRID) -> float:
    """Turn snow layers on top of the first ice layer into ice once refreezing in this step has brought them to pore
    close-off. Mass conserved (height reduced to ice density, liquid water kept in mass). Returns the converted mass
    [m w.e.]."""
    j = -1
    for i in range(GRID.number_nodes):
        if GRID.get_node_ice_fraction(i) >= icf_ice_layer:
            j = i
            break
    converted = 0.0
    k = j - 1
    while k >= 0:
        icf = GRID.get_node_ice_fraction(k)
        if icf < phi_ice_closeoff - 1e-6 or GRID.get_node_refreeze(k) <= 0.0:
            break
        h = GRID.get_node_height(k)
        w = GRID.get_node_liquid_water_content(k) * h
        h_new = icf * h                                    # same ice mass at ice fraction 1
        GRID.set_node_ice_fraction(k, 1.0)
        GRID.set_node_height(k, h_new)
        GRID.set_node_liquid_water_content(k, min(w / h_new, 1.0))
        converted += icf * ice_density * h / water_density
        k -= 1
    return converted

@njit
def check_oob(ice_fraction: float, lwc: float):
    """Check layer mass is conserved.

    Args:
        ice_fraction: Layer's volumetric ice fraction.
        lwc: Layer's liquid water content.

    Raises:
        ValueError: Ice fraction out of bounds.
        ValueError: Liquid water content out of bounds.
    """

    if not 0.0 <= ice_fraction <= 1.0:
        raise ValueError("Ice fraction OOB")
    elif not 0.0 <= lwc <= 1.0:
        raise ValueError("LWC OOB")


@njit
def refreezing(GRID) -> float:
    """Refreeze water in layers.

    This approach is adapted from Bartelt & Lehning (2002).

    .. math::

        \\Delta\\theta_{i} &= -\\Delta\\theta_{w}\\frac{
            \\rho_{w}
                }{
            \\rho_{i}
            }

        \\Delta T &= \\frac{
            c_{w} \\rho_{w} \\Delta\\theta_{w}
                }{
            c_{i} \\rho_{i} \\theta_{i} + c_{w} \\rho_{w} \\theta_{w}
            }

        \\theta_{refrozen} &= \\Delta\\theta_{w} h

    For the maximum water available for refreezing, the latent energy
    release from refreezing water equals the layer warming of the
    layer's ice content, the newly frozen water, and the remaining water
    that cannot be refrozen:

    .. math::
        Q_{refreeze} &= \\theta_{i}
            + \\Delta\\theta_{i}
            + (\\theta_{w} - \\Delta\\theta_{w})

        \\Delta\\theta_{w} L_{f} \\rho_{w} &= \\Delta T_{max}
            \\left [
                (c_{i} \\rho_{i}(\\theta_{i} + \\Delta\\theta_{i}))
                + (c_{w} \\rho_{w} (\\theta_{w} - \\Delta\\theta_{w}))
            \\right ]

        \\Delta\\theta_{w} L_{f} \\rho_{w} &= \\Delta T_{max}
            \\left [
                \\left (
                    c_{i} \\rho_{i}
                    \\left (
                        \\theta_{i} + \\frac{
                            \\rho_{w}
                                }{
                            \\rho_{i}
                            }\\Delta\\theta_{w}
                    \\right )
                \\right )
                + (c_{i} \\rho_{w}(\\theta_{w} - \\Delta\\theta_{w}))
            \\right ]

    Re-arranged in terms of :math:`\\Delta\\theta_{w}`, limited by the
    maximum cold content:

    .. math::

        \\Delta\\theta_{{w}_{max}} = \\frac{
            -\\Delta T_{max}(\\rho_{i}c_{i}\\theta_{i}
            + \\rho_{w}c_{w}\\theta_{w})
                }{
            \\rho_{w}(L_{f}-\\Delta T_{max}(c_{i}-c_{w}))
            }

    .. note::

        The units for :math:`\\Delta\\theta_{w} h` cancel out to m w.e.
        as long as the density of waer is set to 1000 kg m^-3.
        Note that :math:`\\Delta\\theta_{i} h` is in m **ice**
        equivalent, but both the refreeze parameter and returned
        refrozen water are in m w.e.

    Args:
        GRID (Grid): Glacier data structure.

    Returns:
        Refrozen water, [|m w.e.|].
    """

    # Maximum snow fractional ice content:
    phi_ice_max = (pore_closeoff_density - air_density) / (
        ice_density - air_density
    )
    ice_water_density_ratio = ice_density / water_density
    ice_spec_density_product = spec_heat_ice * ice_density
    water_heat_density_product = water_density * lat_heat_melting
    refrozen_water = 0.0
    for idx in range(GRID.number_nodes):
        if (GRID.get_node_temperature(idx) <= zero_temperature) & (
            GRID.get_node_liquid_water_content(idx) > 0.0
        ):
            ice_fraction = GRID.get_node_ice_fraction(idx)
            lwc = GRID.get_node_liquid_water_content(idx)
            temperature = GRID.get_node_temperature(idx)
            dT_max = temperature - zero_temperature

            # Volumetric/density limit
            dtheta_w_max_density = max(
                (phi_ice_max - ice_fraction) * ice_water_density_ratio, 0.0
            )

            # Cold content limit, dT_max is negative
            dtheta_w_max_coldcontent = -(
                dT_max
                * (
                    (ice_fraction * ice_spec_density_product)
                    + (lwc * water_density * spec_heat_water)
                )
            ) / (
                water_density
                * (
                    lat_heat_melting
                    - dT_max * (spec_heat_ice - spec_heat_water)
                )
            )
            dtheta_w_bulk = (
                -(ice_spec_density_product * ice_fraction * dT_max)
                / water_heat_density_product
            )

            # Maximum refrozen water, dtheta_w >= 0.0
            dtheta_w = min(
                (
                    lwc,
                    dtheta_w_max_density,
                    dtheta_w_max_coldcontent,
                    dtheta_w_bulk,
                )
            )

            # Change in the layer's ice fraction
            dtheta_i = (water_density / ice_density) * dtheta_w

            # Update ice fraction and liquid water content
            check_oob(ice_fraction=ice_fraction + dtheta_i, lwc=lwc - dtheta_w)
            GRID.set_node_liquid_water_content(idx, lwc - dtheta_w)
            GRID.set_node_ice_fraction(idx, ice_fraction + dtheta_i)

            # Layer temperature change
            dT = (dtheta_w * water_heat_density_product) / (
                (ice_spec_density_product * GRID.get_node_ice_fraction(idx))
                + (
                    spec_heat_water
                    * water_density
                    * (GRID.get_node_liquid_water_content(idx))
                )
            )
            GRID.set_node_temperature(idx, temperature + dT)
            if temperature + dT < 0.0:
                raise ValueError("Temperature OOB")
        else:
            dtheta_i = 0.0
            dtheta_w = 0.0
        height = GRID.get_node_height(idx)

        # Set refreezing
        # dtheta_i * ice_density = dtheta_w * water_density
        delta_mwe = dtheta_w * height
        GRID.set_node_refreeze(idx, delta_mwe)
        refrozen_water += delta_mwe

    return refrozen_water
