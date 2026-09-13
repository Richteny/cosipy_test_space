import numpy as np
from numba import njit

from cosipy.constants import Constants

# only required for njitted functions
densification_method = Constants.densification_method
snow_ice_threshold = Constants.snow_ice_threshold
minimum_snow_layer_height = Constants.minimum_snow_layer_height
zero_temperature = Constants.zero_temperature
ice_density = Constants.ice_density
rho_firn = Constants.rho_firn #500.0 (Kronenberg et al. 2022)
rho_max_dm = Constants.rho_max_dm #300.0 (calibrated for Abramov, EBFM default 175.0)

def densification(GRID, SLOPE, dt, accumulation):
    """Apply densification to the snowpack.

    Implemented densification methods:

        - **Boone**: Densification through overburden pressure. Essery
          et al. (2013).
        - **Vionnet**: Densification through overburden stress. Vionnet
          et al. (2011).
        - **Ligtenberg2011**: firn densification after Ligtenberg 2011 (EBFM)
        - **Ligtenberg-vanKampenhout**: combines Ligtenberg scheme with new vanKampenhout snow metamorphism, as used in the EBFM model
        - **empirical**: Empirical compaction with constant time scale.
        - **constant**: Constant density (no compaction).

    Args:
        GRID (Grid): Glacier data structure.
        SLOPE (np.ndarray): Slope of the surface [|degree|].
        dt (int): Integration time [s].

    Raises:
        NotImplementedError: Densification method is not allowed.
    """
    densification_allowed = ['Boone', 'Vionnet', 'empirical', 'constant', 'Ligtenberg11', 'Ligtenberg-vanKampenhout']
    if densification_method == 'Boone':
        method_Boone(GRID,SLOPE,dt)
    elif densification_method == 'Vionnet':
        method_Vionnet(GRID,SLOPE,dt)
    elif densification_method == 'empirical':
        method_empirical(GRID,SLOPE,dt)
    elif densification_method == "Ligtenberg11":
        method_Ligtenberg(GRID, dt, accumulation, split=False)
    elif densification_method == "Ligtenberg-vanKampenhout":
        # rho < rho_firn -> van Kampenhout et al. 2017
        # rho >= rho_firn -> Arthern 2010 + Ligtenberg 2011
        method_vanKampenhout(GRID, dt)
        method_Ligtenberg(GRID, dt, accumulation, split=True)
    elif densification_method == 'constant':
        pass
    else:
        error_msg = (
            f'Densification method = "{densification_method}"',
            "is not allowed, must be one of",
            f'{", ".join(densification_allowed)}',
        )
        raise ValueError(" ".join(error_msg))


@njit
def log_fraction_warning(grid, idx: int, dtheta_i: float, dtheta_w: float):
    """Warn user if liquid fractions are physically impossible.
    
    Args:
        grid (Grid): Glacier data structure.
        idx: Node index.
        dtheta_i: Change in volumetric ice fraction.
        dtheta_w: Change in volumetric liquid water content.
    """
    if (
        grid.get_node_ice_fraction(idx)
        + grid.get_node_liquid_water_content(idx)
        + grid.get_node_porosity(idx)
        > 1.0
    ):
        ice_fraction = (1 + dtheta_i) * grid.get_node_ice_fraction(idx)
        lwc_fraction = (1 + dtheta_w) * grid.get_node_liquid_water_content(idx)
        porosity = grid.get_node_porosity(idx)
        print(ice_fraction, lwc_fraction, porosity)
        print(ice_fraction + lwc_fraction + porosity)
        print(
            "Fraction > 1: ",
            (
                grid.get_node_ice_fraction(idx)
                + grid.get_node_liquid_water_content(idx)
                + porosity
            ),
        )


@njit
def copy_layer_profiles(GRID) -> tuple:
    """Get a copy of the layer profiles.

    `np.array` returns a copy by default and is 2x faster than np.copy
    (which is not supported by numba).

    Args:
        GRID (Grid): Glacier data structure.

    Returns:
        Profiles for height, density, temperature, liquid water content,
        and ice fraction.
    """

    heights = np.array(GRID.get_height())
    densities = np.array(GRID.get_density())
    temperatures = np.array(GRID.get_temperature())
    lwcs = np.array(GRID.get_liquid_water_content())
    ice_fractions = np.array(GRID.get_ice_fraction())

    return heights, densities, temperatures, lwcs, ice_fractions

@njit
def method_Boone(GRID, SLOPE, dt):
    """Densification through overburden pressure.

    After Essery et al., (2013).

    Args:
        GRID (Grid): Glacier data structure.
        SLOPE (np.ndarray): Slope of the surface [|degree|].
        dt (int): Integration time [s].
    """

    # Constants
    c1 = 2.8e-6
    c2 = 0.042
    c3 = 0.046
    c4 = 0.081
    c5 = 0.018
    eta0 = 3.7e7
    rho0 = 150

    # Overburden snow mass
    M_s = 0.0

    # Get copy of layer heights and layer densities
    height, rho, t, lwc, icf = copy_layer_profiles(GRID)

    # Loop over all internal snow nodes
    for idxNode in range(0,GRID.get_number_snow_layers() , 1):

        if (rho[idxNode] < snow_ice_threshold) & (height[idxNode] > minimum_snow_layer_height):

            # Get overburden snow mass
            if idxNode>0:
                M_s = M_s + rho[idxNode-1]*height[idxNode-1]
            elif idxNode==0:
                M_s = M_s + rho[0]*(height[0]/2.0)

            # Viscosity
            eta = eta0 * np.exp(c4*(zero_temperature-t[idxNode])+c5*rho[idxNode])

            # Rate of change in the density
            dRho = (((M_s*9.81)/eta) + c1*np.exp(-c2*(zero_temperature-t[idxNode]) - c3*np.maximum(0.0,rho[idxNode]-rho0)))*dt
           
            # Calc changes in volumetric fractions of ice and water
            # No water in layer
            if lwc[idxNode] == 0.0:
                dtheta_i = dRho
                dtheta_w = 0.0
            # layer contains water
            else:
                dtheta_i = (dRho/2.0)
                dtheta_w = (dRho/2.0)
            
            # Set new fractions
            GRID.set_node_ice_fraction(idxNode, (1+dtheta_i) * icf[idxNode])
            GRID.set_node_liquid_water_content(idxNode, (1+dtheta_w) * lwc[idxNode])

            # Set new layer height (compaction)
            GRID.set_node_height(idxNode, (1-dRho) * height[idxNode])

            log_fraction_warning(GRID, idxNode, dtheta_i, dtheta_w)



def method_Vionnet(GRID, SLOPE, dt):
    """Densification through overburden stress.

    After Vionnet et al., (2011).

    Args:
        GRID (Grid): Glacier data structure.
        SLOPE (np.ndarray): Slope of the surface [|degree|].
        dt (int): Integration time [s].
    """

    # Constants
    f2 = 1.0
    eta0 = 7.62237e6  # [N s m^-2]
    a = 0.1           # [K^-1] 
    b = 0.023         # [m^3 kg^-1]
    c = 250           # [kg m^-3]

    # Vertical Stress 
    sigma = 0.0

    # Get copy of layer heights and layer densities
    height, rho, t, lwc, icf = copy_layer_profiles(GRID)

    # Loop over all internal snow nodes
    for idxNode in range(0,GRID.get_number_snow_layers() , 1):

        if (rho[idxNode] < snow_ice_threshold) & (height[idxNode] > minimum_snow_layer_height):

            # Parameter f1
            f1 = 1 / (1+60.0*(lwc[idxNode]/height[idxNode]))
          
            # Snow viscosity
            eta = f1*f2*eta0*(rho[idxNode]/c)*np.exp(a*(273.14-t[idxNode])+b*rho[idxNode])

            # Vertical stress
            if idxNode>0:
                sigma = sigma + 9.81*np.cos(SLOPE)*rho[idxNode-1]*height[idxNode-1]
            elif idxNode==0:
                # Take only half layer height of the first layer
                sigma = sigma + 9.81*np.cos(SLOPE)*rho[0]*(height[0]/2.0)

            # Rate of change for the layer height
            dD = (-sigma/eta)*dt 

            # Rate of change for the density
            # dRho = dD*rho[idxNode]
            
            # Calc changes in volumetric fractions of ice and water
            # No water in layer
            if lwc[idxNode]==0.0:
                dtheta_i = -dD
                dtheta_w = 0.0
            # layer contains water
            else:
                dtheta_i = -dD/2.0
                dtheta_w = -dD/2.0

            # Set new volumetric fractions
            GRID.set_node_ice_fraction(idxNode, (1+dtheta_i) * icf[idxNode])
            GRID.set_node_liquid_water_content(idxNode, (1+dtheta_w) * lwc[idxNode])

            # Set new layer height (compaction)
            GRID.set_node_height(idxNode, (1+dD)*height[idxNode])

            log_fraction_warning(GRID, idxNode, dtheta_i, dtheta_w)


def method_empirical(GRID, SLOPE, dt):
    """Empirical snow compaction using a constant time scale.
    
    Args:
        GRID (Grid): Glacier data structure.
        SLOPE (np.ndarray): Slope of the surface [|degree|].
        dt (int): Integration time [s].
    """

    rho_max = 600.0  # maximum attainable density [kg m^-3]
    #tau = 3.6e5  # empirical compaction time scale [s]
    tau = 8.0e5  # empirical compaction time scale [s]
    
    # Get copy of layer heights and layer densities
    rho = np.array(GRID.get_density())

    # Loop over all internal snow nodes
    for idxNode in range(0,GRID.get_number_snow_layers() , 1):

        # Rate of density change
        dRho = (1/tau) * (rho_max-GRID.get_node_density(idxNode)) * dt

        if (1 - (dRho / GRID.get_node_density(idxNode))) < 1:
            # Set the new ice fraction
            GRID.set_node_ice_fraction(idxNode, (rho_max + (rho[idxNode]-rho_max) * np.exp(-dt/tau))/Constants.ice_density )

            # Set height change
            GRID.set_node_height(idxNode, (1-(dRho/GRID.get_node_density(idxNode)))*GRID.get_node_height(idxNode))

@njit
def _apply_compaction(GRID, rho, drho, h, icf, lwc):
    """ Mass consreving compaction.
    """
    ratio = np.ones_like(rho)
    for i in range(len(drho)):
        if drho[i] > 0.0:
            ratio[i] = rho[i] / (rho[i] + drho[i])
            GRID.set_node_height(i, h[i] * ratio[i])
            GRID.set_node_ice_fraction(i, icf[i] / ratio[i])
            GRID.set_node_liquid_water_content(i, lwc[i] / ratio[i])
    return ratio


@njit
def method_vanKampenhout(GRID, dt):
    """Snow settling, compaction and metamorphosis after van Kampenhout et al. (2017).
    """

    cdm3, cdm4 = 2.777e-6, 0.04           # s-1, K-1
    eta0, a_eta, b_eta, c_eta = 7.62237e6, 0.1, 0.023, 358.0

    rho = np.asarray(GRID.get_density())
    h = np.asarray(GRID.get_height())
    T = np.asarray(GRID.get_temperature())
    lwc = np.asarray(GRID.get_liquid_water_content())
    icf = np.asarray(GRID.get_ice_fraction())

    cond = rho < rho_firn

    # --- Gl. 2: zerstoerende Metamorphose ---
    cdm1 = np.where(rho < rho_max_dm, 1.0, np.exp(-0.046 * (rho - rho_max_dm)))
    cdm2 = np.where(lwc == 0.0, 1.0, 2.0)
    drho_dm = np.where(cond, dt * rho * cdm3 * cdm2 * cdm1
                       * np.exp(-cdm4 * (zero_temperature - T)), 0.0)

    rho_1 = np.minimum(rho + drho_dm, ice_density)
    ratio = _apply_compaction(GRID, rho, rho_1 - rho, h, icf, lwc)

    # --- Gl. 3/4: Auflastdruck, aus der aktualisierten Dichte ---
    h = h * ratio
    lwc = lwc / ratio
    icf = icf / ratio 
    #h = np.asarray(GRID.get_height())
    #lwc = np.asarray(GRID.get_liquid_water_content())
    #icf = np.asarray(GRID.get_ice_fraction())

    f1 = 1.0 / (1.0 + 60.0 * lwc)          # EBFM: subW/(Dwater*subZ) == COSIPYs lwc
    eta = f1 * 4.0 * eta0 * (rho_1 / c_eta) \
        * np.exp(a_eta * (zero_temperature - T) + b_eta * rho_1)
    Psload = np.cumsum(rho_1 * h) - 0.5 * rho_1 * h

    drho_ob = np.where(cond, dt * rho_1 * Psload / eta, 0.0)
    rho_2 = np.minimum(rho_1 + drho_ob, ice_density)
    _apply_compaction(GRID, rho_1, rho_2 - rho_1, h, icf, lwc)


@njit
def method_Ligtenberg(GRID, dt, accumulation, split):
    """ Densification based on in situ measurements of Antarctic snow compaction (used in EBFM)
        after  Arthern et al. (2010) and modified by Ligtenberg et al. (2011).
    
        Parameters:
                   dt = integration time in a model time-step [s]
        Input: 
                   GRID = subsurface grid variables
                   accumulation = grid annual accumulation [m a-1]
        Output rho (z) = layer_density
    """

    R, Ec, Eg, g = 8.314, 60e3, 42.4e3, 9.81

    rho = np.asarray(GRID.get_density())
    h = np.asarray(GRID.get_height())
    T = np.asarray(GRID.get_temperature())
    T_avg = np.asarray(GRID.get_average_temperature())
    lwc = np.asarray(GRID.get_liquid_water_content())
    icf = np.asarray(GRID.get_ice_fraction())

    b = max(accumulation * 1000.0, 1.0)           # mm w.e. a-1
    dt_frac = dt / (365.0 * 24.0 * 3600.0)

    if split:
        cond = (rho >= rho_firn) & (rho < snow_ice_threshold)
    else:
        cond = rho < snow_ice_threshold

    C = np.where(rho < 550.0,
                 0.07 * max(1.435 - 0.151 * np.log(b), 0.25),
                 0.03 * max(2.366 - 0.293 * np.log(b), 0.25))

    # Estimate average temperature using an Exponential Moving Average (EMA)
    alpha = (2 * dt_frac) / (5 + dt_frac) #smoothing factor (five-yearly average)
    T_avg = (T * alpha) + (T_avg * (1.0 - alpha))
    GRID.set_average_temperature(T_avg)

    drho = np.where(cond,
                    dt_frac * C * b * g * np.maximum(ice_density - rho, 0.0)
                    * np.exp((-Ec / (R * T)) + (Eg / (R * T_avg))), 0.0)

    _apply_compaction(GRID, rho, drho, h, icf, lwc)

