import numpy as np
from numba import njit
from cosipy.constants import Constants
from cosipy.modules.refreezing import interface_refreeze_limit, add_refrozen_to_ice

preferential_percolation_method = Constants.preferential_percolation_method
preferential_percolation_depth = Constants.preferential_percolation_depth

snow_ice_threshold = Constants.snow_ice_threshold
ice_density = Constants.ice_density
air_density = Constants.air_density
icf_ice = (snow_ice_threshold - air_density) / (ice_density - air_density) #same as in refreezing

@njit
def percolation(GRID, water: float, dt: int, interface_water: float = 0.0) -> tuple:
    """Percolate melt water through the snow- and firn pack.

    Superimposed ice (interface_water > 0, see refreezing.py): water that reaches the first ice layer after passing
    through snow refreezes onto it: at most interface_water [m w.e.] (water that came through or from snow in this
    step), the layer's cold content and the heat conducted into it over dt (refreezing.interface_refreeze_limit).
    The rest runs off as before. If the snow melted completely in this step, the first ice layer is the top layer
    and the water released from the snow refreezes there.

    Surface water is first placed in the column, either entirely in the top layer ('disabled') or distributed over depth following Marchenko et al. (2017). Whatever exceeds the irreducible water content is then routed downwards with the bucket method.
    Bucket method (Bartelt & Lehning, 2002).

    Args:
        GRID (Grid): Glacier data structure.
        water: Melt water at the surface, [|m w.e.| q.].
        dt: Integration time [s].

    Returns:
        Runoff [m w.e.] and water refrozen at the snow-ice interface [m w.e.].
    """
    refrozen_interface = 0.0

    # convert m to mm = kg/m2, not needed because of change to fraction
    # water = water * 1000
    if water > 0.0:
        if preferential_percolation_method == "Marchenko17":
            method_Marchenko(GRID, water)
        else:
        
            # convert kg/m2 to kg/m3
            water = water / GRID.get_node_height(0)
            # kg/m3 to fraction
            # water = water / 1000

            # set liquid water of top layer (idx, LWCnew) in m
            GRID.set_node_liquid_water_content(
                0, GRID.get_node_liquid_water_content(0) + float(water)
            )

    # snow melted away in this step: its water sits in the top layer, which is ice
    if interface_water > 0.0 and GRID.get_node_ice_fraction(0) >= icf_ice:
        w0 = GRID.get_node_liquid_water_content(0) * GRID.get_node_height(0)
        r = min(w0, interface_water, interface_refreeze_limit(GRID, 0, dt))
        if r > 0.0:
            GRID.set_node_liquid_water_content(0, (w0 - r) / GRID.get_node_height(0))
            add_refrozen_to_ice(GRID, 0, r)
            refrozen_interface += r
            interface_water -= r
        interface_water = 0.0                      # only the top interface

    # Loop over all internal grid points for percolation
    for idxNode in range(0, GRID.number_nodes - 1, 1):
        theta_e = GRID.get_node_irreducible_water_content(idxNode)
        theta_w = GRID.get_node_liquid_water_content(idxNode)

        # Residual volume fraction of water (m^3 which is equal to m)
        residual = np.maximum((theta_w - theta_e), 0.0)

        if residual > 0.0:
            GRID.set_node_liquid_water_content(idxNode, theta_e)

            """
            old:
            GRID.set_node_liquid_water_content(
                idxNode + 1,
                GRID.get_node_liquid_water_content(idxNode + 1) + residual,
            )

            new:
            If water is pushed to next layer, the layer heights have to
            be considered because of fractions.
            """
            residual = residual * GRID.get_node_height(idxNode)
            # snow-ice interface: refreeze onto the first ice layer (superimposed ice)
            if interface_water > 0.0 and GRID.get_node_ice_fraction(idxNode) < icf_ice \
                    and GRID.get_node_ice_fraction(idxNode + 1) >= icf_ice:
                r = min(residual, interface_water, interface_refreeze_limit(GRID, idxNode + 1, dt))
                if r > 0.0:
                    add_refrozen_to_ice(GRID, idxNode + 1, r)
                    residual -= r
                    refrozen_interface += r
                interface_water = 0.0                  # only the top interface
            GRID.set_node_liquid_water_content(
                idxNode + 1,
                GRID.get_node_liquid_water_content(idxNode + 1)
                + residual / GRID.get_node_height(idxNode + 1),
            )

    Q = get_runoff(GRID)

    return Q, refrozen_interface

@njit
def method_Marchenko(GRID, surface_water: float):
    """Statistical preferential percolation scheme (Gaussian) after Marchenko et al. (2017).
    Water is distributed instantaneously following a half-normal PDF with sigma = zlim/3,
    but only over the porous column above the first ice layer (snow/firn that can hold or
    refreeze water). In deep firn this is identical to the original scheme; over thin snow on
    ice the water stays in the snow instead of being placed in ice layers (immediate runoff)."""

    h = np.asarray(GRID.get_height(), dtype=np.float64)
    lwc = np.asarray(GRID.get_liquid_water_content(), dtype=np.float64)
    icf = np.asarray(GRID.get_ice_fraction(), dtype=np.float64)

    # number of porous layers above the first ice layer
    n_por = len(h)
    for i in range(len(h)):
        if icf[i] >= icf_ice:
            n_por = i
            break

    if n_por == 0:
        # bare ice at the surface: all water to the top layer (bucket routes it to runoff)
        GRID.set_node_liquid_water_content(0, lwc[0] + surface_water / h[0])
        return

    z = np.cumsum(h) - 0.5 * h
    sigma = preferential_percolation_depth / 3.0
    PDF_normal = 2.0 * (np.exp(-(z**2) / (2.0 * sigma**2))
                        / (sigma * np.sqrt(2.0 * np.pi)))
    PDF_normal_height = PDF_normal * h
    PDF_normal_height[n_por:] = 0.0                  # no water placed in or below the first ice layer
    total = np.sum(PDF_normal_height)
    if total > 0.0:
        water = lwc + (PDF_normal_height / total * surface_water) / h
        GRID.set_liquid_water_content(water)
    else:
        GRID.set_node_liquid_water_content(0, lwc[0] + surface_water / h[0])

#OLD VERSION, copy from FRICOSIPY / EBFM
"""
@njit
def method_Marchenko(GRID, surface_water: float):
    #Statistical preferential percolation scheme (Gaussian) after Marchenko et al. (2017).
    #Water is distributed instantenously over the column following a normal PDF with sigma = zlim/3.

    h = np.asarray(GRID.get_height(), dtype=np.float64)
    lwc = np.asarray(GRID.get_liquid_water_content(), dtype=np.float64)

    # calculate layer depth -> get_depth returns list, so calculate directly from height to be numba friendly
    z = np.cumsum(h) - 0.5*h

    sigma = preferential_percolation_depth / 3.0
    PDF_normal = 2.0 * (np.exp(-(z**2) / (2.0 * sigma**2))
                        / (sigma * np.sqrt(2.0 * np.pi)))
    PDF_normal_height = PDF_normal * h
    total = np.sum(PDF_normal_height)
    if total > 0.0:
        Normalise = PDF_normal_height / total
        water = lwc + (Normalise * surface_water) / h
        GRID.set_liquid_water_content(water)
    else:
        #column deeper than percolation window
        #add everything to top
        GRID.set_node_liquid_water_content(0, GRID.get_node_liquid_water_content(0) + surface_water / h[0]
        )
"""

@njit
def get_runoff(grid) -> float:
    """Get meltwater runoff for a column.

    Runoff is equal to LWC in the last node & must be converted
    from kg/m3 to kg/m2. Converting from fraction to kg/m3 (\\*1000) and
    from mm to m (/1000) is unnecessary.

    Args:
        grid (Grid): Glacier data structure.

    Returns:
        Meltwater runoff.
    """

    max_index = grid.number_nodes - 1
    runoff = grid.get_node_liquid_water_content(
        max_index
    ) * grid.get_node_height(max_index)
    grid.set_node_liquid_water_content(max_index, 0.0)

    return runoff
