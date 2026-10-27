import numpy as np
import pandas as pd

from cosipy.config import Config
from cosipy.constants import Constants
from cosipy.cpkernel.init import init_snowpack, load_snowpack
from cosipy.cpkernel.io import IOClass
from cosipy.modules.albedo import updateAlbedo, albedo_memory
from cosipy.modules.densification import densification
from cosipy.modules.evaluation import evaluate, resample_output, create_tsl_df, eval_tsl, resample_by_hand
from cosipy.modules.heatEquation import solveHeatEquation
from cosipy.modules.penetratingRadiation import penetrating_radiation
from cosipy.modules.percolation import percolation
from cosipy.modules.refreezing import refreezing
from cosipy.modules.roughness import updateRoughness
from cosipy.modules.surfaceTemperature import update_surface_temperature
from cosipy.modules.surfaceTemperature import method_EW_Sonntag


# Depth zones (m) for the near-surface liquid water outputs, for comparison with Sentinel-1 wet-snow maps.
# C-band reaches only a few cm into wet snow, but metres into dry snow above a wet layer, so the depth that
# matters is not known in advance: the maximum liquid water content is stored per zone (no threshold applied),
# and depth and threshold are chosen in the analysis (max over the zones down to the chosen depth).
# The deepest zone ends at preferential_percolation_depth, the depth down to which the Marchenko et al. (2017)
# scheme places surface water directly.
LWC_ZONES = {"LWC_00_10": (0.0, 0.1), "LWC_10_50": (0.1, 0.5),
             "LWC_DEEP": (0.5, float(Constants.preferential_percolation_depth))}


def zone_max_lwc(height: np.ndarray, lwc: np.ndarray, z_top: float, z_bottom: float) -> float:
    """Highest liquid water volume fraction of the layers overlapping the depth zone [z_top, z_bottom).

    Args:
        height: Layer thicknesses from the surface down [|m|].
        lwc: Layer liquid water content, volume fraction [-].
        z_top: Upper limit of the zone, depth below the surface [|m|].
        z_bottom: Lower limit of the zone, depth below the surface [|m|].

    Returns:
        Maximum liquid water volume fraction in the zone [-] (NaN if no layer reaches the zone).
    """
    bottom = np.cumsum(height)                        # depth of each layer's lower boundary
    top = bottom - height                             # depth of each layer's upper boundary
    inside = (top < z_bottom) & (bottom > z_top)
    return float(np.max(lwc[inside])) if inside.any() else np.nan


def init_nan_array_1d(nt: int) -> np.ndarray:
    """Initialise and fill an array with NaNs.
    
    Args:
        nt: Array size (time dimension).

    Returns:
        NaN array.
    """
    if not Config.WRF_X_CSPY:
        x = np.full(nt, np.nan)
    else:
        x = None

    return x


def init_nan_array_2d(nt: int, max_layers: int) -> np.ndarray:
    """Initialise and fill an array with NaNs.
    
    Args:
        nt: Array's temporal resolution.
        max_layers: Array's spatial resolution.

    Returns:
        2D NaN array.
    """
    if not Config.WRF_X_CSPY and Config.full_field:
        x = np.full((nt, max_layers), np.nan)
    else:
        x = None

    return x

def calc_wetbulb_temperature(TK: float, RH: float, pressure: float) -> float:
    """Compute wetbulb temperature in °C

    Args:
        TK: Air temperature in K
        RH: relative humidity in percent
        pressure: Air pressure in hPa
   Returns:
       Wet-bulb temperature in °C
    """
    #values taken from T&C and Ding et al. 2014
    Ta = TK - 273.16
    esat = method_EW_Sonntag(TK)
    ea = (RH / 100.0) * esat
    
    lheat = 1000.0 * (2501.3 - 2.361 * Ta)    # J/kg
    cp = 1005.0 + ((Ta + 23.15)**2) / 3364.0  # J/kg/K
    Pre_Pa = pressure * 100.0                 # Pa
    gam = cp * Pre_Pa / (0.622 * lheat)

    esat_Pa = esat * 100.0
    ea_Pa = ea * 100.0
    del_ = (4098.0 * esat_Pa) / ((237.3 + Ta)**2)

    Twb = Ta - (esat_Pa - ea_Pa) / (gam + del_)
    return Twb

def partition_precipitation(TK: float, Twb: float, RH: float, pressure: float, RRR: float, Zref: float, phase_offset: float) -> tuple:
    """
    Compute snow/rain fractions of total precipitation.
    Args:
        Ta: Air temperature [°C]
        Twb: Wet-bulb temperature [°C]
        RH_percent: Relative humidity [%]
        PRES_hPa: Pressure [hPa]
        Pr: Total precipitation [mm w.e.]
    Returns:
        Pr_liq, Pr_sno (rain/snow in mm w.e.)
    """
    Ta = TK - 273.16
    # constants
    g = 9.81
    P_Ref = 1013.25
    Rd = 287.05

    # reference elevation in km
    #Zref = -((Ta + 15.0)/2 + 273.15) * (Rd / g) * np.log(pressure / P_Ref)
    #calc above from T&C, but not in original paper
    Zref = Zref / 1000.0

    # relative humidity fraction
    rh_frac = RH / 100.0

    # threshold parameters
    dT_ = 0.215 - 0.099 * rh_frac + 1.018 * rh_frac**2
    dS_ = 2.374 - 1.634 * rh_frac
    T0_ = -5.87 - 0.1042 * Zref + 0.0885 * Zref**2 + 16.06 * rh_frac - 9.614 * rh_frac**2

    #tuning param
    T0_ = T0_ + phase_offset

    Tmin, Tmax = T0_, T0_
    if dT_ / dS_ > np.log(2):
        Tmin = T0_ - dS_ * np.log(np.exp(dT_ / dS_) - 2.0 * np.exp(-dT_ / dS_))
        Tmax = 2.0 * T0_ - Tmin

    solid_fraction = 1.0 / (1.0 + np.exp((Twb - T0_) / dS_))

    # initialize
    Pr_liq, Pr_sno = 0.0, 0.0

    if Tmin < Twb < Tmax:
        Pr_sno = RRR * solid_fraction
        Pr_liq = RRR * (1.0 - solid_fraction)
    elif Twb <= Tmin:
        Pr_sno = RRR
        Pr_liq = 0.0
    else:
        Pr_sno = 0.0
        Pr_liq = RRR

    return Pr_liq, Pr_sno



def cosipy_core(DATA, indY, indX, GRID_RESTART=None, stake_names=None, stake_data=None, opt_dict=None):
    """Cosipy core function.

    The calculations are performed on a single core.

    Args:
        DATA (xarray.Dataset): Dataset with single grid point.
        indY (int): The grid cell's Y index.
        indX (int): The grid cell's X index.
        GRID_RESTART (xarray.Dataset): Use a restart dataset instead of
            creating an initial profile. Default ``None``.
        stake_names (list): Stake names. Default ``None``.
        stake_data (pd.Dataframe): Stake measurements. Default ``None``.

    Returns:
        All calculated variables for one grid point.
    """

    # Declare locally for faster lookup
    dt = Constants.dt
    max_layers = Constants.max_layers
    z = Constants.z
    mult_factor_RRR = Constants.mult_factor_RRR
    rrr_factor_summer = Constants.rrr_factor_summer
    rrr_factor_winter = Constants.rrr_factor_winter
    densification_method = Constants.densification_method
    precippartition_method = Constants.precip_partitioning_method
    ice_density = Constants.ice_density
    water_density = Constants.water_density
    minimum_snowfall = Constants.minimum_snowfall
    zero_temperature = Constants.zero_temperature
    lat_heat_sublimation = Constants.lat_heat_sublimation
    lat_heat_melting = Constants.lat_heat_melting
    lat_heat_vaporize = Constants.lat_heat_vaporize
    center_snow_transfer_function = Constants.center_snow_transfer_function
    spread_snow_transfer_function = Constants.spread_snow_transfer_function
    constant_density = Constants.constant_density
    albedo_fresh_snow = Constants.albedo_fresh_snow
    albedo_firn = Constants.albedo_firn
    WRF_X_CSPY = Config.WRF_X_CSPY
    bias_LWIN = Constants.bias_LWin
    mult_factor_WS = Constants.mult_factor_WS
    bias_T2 = Constants.bias_T2
    # Seasonal T2 bias (JJAS / Oct-May): when both are set they replace bias_T2 (constants.toml or opt_dict[21:23])
    bias_T2_summer = getattr(Constants, "bias_T2_summer", None)
    bias_T2_winter = getattr(Constants, "bias_T2_winter", None)
    # Replace values from constants.py if coupled
    # TODO: This only affects the current module scope instead of global.
    if WRF_X_CSPY:
        dt = int(DATA.DT.values)
        max_layers = int(DATA.max_layers.values)
        z = float(DATA.ZLVL.values)

    #del later
    DEBUG_ELEV = None

    # Replace imported variables with content of the opt_dict. If it's empty
    # nothing happens.
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
        bias_LWIN = opt_dict[12]
        mult_factor_WS = opt_dict[13]
        bias_T2 = opt_dict[14]
        t_wet = opt_dict[15]
        t_dry = opt_dict[16]
        t_K = opt_dict[17]
        minimum_snowfall = opt_dict[18]
        rrr_factor_summer = opt_dict[19]
        rrr_factor_winter = opt_dict[20]
        if len(opt_dict) > 22:
            bias_T2_summer = opt_dict[21]
            bias_T2_winter = opt_dict[22]

    nt = len(DATA.time.values)  # accessing DATA is expensive
    """
    Local variables bypass local array creation for WRF_X_CSPY
    TODO: Implement more elegant solution.
    """
    if not WRF_X_CSPY:
        _RRR = init_nan_array_1d(nt)
        _RAIN = init_nan_array_1d(nt)
        _SNOWFALL = init_nan_array_1d(nt)
        _LWin = init_nan_array_1d(nt)
        _LWout = init_nan_array_1d(nt)
        _H = init_nan_array_1d(nt)
        _LE = init_nan_array_1d(nt)
        _B = init_nan_array_1d(nt)
        _QRR = init_nan_array_1d(nt)
        _MB = init_nan_array_1d(nt)
        _surfMB = init_nan_array_1d(nt)
        _MB = init_nan_array_1d(nt)
        _Q = init_nan_array_1d(nt)
        _SNOWHEIGHT = init_nan_array_1d(nt)
        _TOTALHEIGHT = init_nan_array_1d(nt)
        _TS = init_nan_array_1d(nt)
        _ALBEDO = init_nan_array_1d(nt)
        _ME = init_nan_array_1d(nt)
        _intMB = init_nan_array_1d(nt)
        _EVAPORATION = init_nan_array_1d(nt)
        _SUBLIMATION = init_nan_array_1d(nt)
        _CONDENSATION = init_nan_array_1d(nt)
        _DEPOSITION = init_nan_array_1d(nt)
        _REFREEZE = init_nan_array_1d(nt)
        _NLAYERS = init_nan_array_1d(nt)
        _subM = init_nan_array_1d(nt)
        _Z0 = init_nan_array_1d(nt)
        _surfM = init_nan_array_1d(nt)
        _MOL = init_nan_array_1d(nt)
        _new_snow_height = init_nan_array_1d(nt)
        _new_snow_timestamp = init_nan_array_1d(nt)
        _old_snow_timestamp = init_nan_array_1d(nt)
        _CY_SNOWHEIGHT = init_nan_array_1d(nt)
        _LWC_00_10 = init_nan_array_1d(nt)
        _LWC_10_50 = init_nan_array_1d(nt)
        _LWC_DEEP = init_nan_array_1d(nt)

        _LAYER_HEIGHT = init_nan_array_2d(nt, max_layers)
        _LAYER_RHO = init_nan_array_2d(nt, max_layers)
        _LAYER_T = init_nan_array_2d(nt, max_layers)
        _LAYER_LWC = init_nan_array_2d(nt, max_layers)
        _LAYER_CC = init_nan_array_2d(nt, max_layers)
        _LAYER_POROSITY = init_nan_array_2d(nt, max_layers)
        _LAYER_ICE_FRACTION = init_nan_array_2d(nt, max_layers)
        _LAYER_IRREDUCIBLE_WATER = init_nan_array_2d(nt, max_layers)
        _LAYER_REFREEZE = init_nan_array_2d(nt, max_layers)
        _LAYER_TAVG = init_nan_array_2d(nt, max_layers)
        _LAYER_HY = init_nan_array_2d(nt, max_layers)

    #--------------------------------------------
    # Initialize snowpack or load restart grid
    #--------------------------------------------
    if GRID_RESTART is None:
        GRID = init_snowpack(DATA)
    else:
        GRID = load_snowpack(GRID_RESTART)

    # Create the local output datasets if not coupled
    RESTART = None
    if not WRF_X_CSPY:
        IO = IOClass(DATA, opt_dict)
        RESTART = IO.create_local_restart_dataset()

    # hours since the last snowfall (albedo module)
    # hours_since_snowfall = 0

    #--------------------------------------------
    # Get data from file
    #--------------------------------------------
    T2_raw = DATA.T2.values.copy()
    _months = pd.DatetimeIndex(DATA.time.values).month.values
    # T2 bias per time step: seasonal (JJAS / Oct-May, same months as rrr_factor_summer / _winter) when both seasonal
    # values are set, else bias_T2 all year
    _seasonal_t2 = (bias_T2_summer is not None) and (bias_T2_winter is not None) \
        and np.isfinite(float(bias_T2_summer)) and np.isfinite(float(bias_T2_winter))
    if _seasonal_t2:
        bias_T2_step = np.where(np.isin(_months, (6, 7, 8, 9)), float(bias_T2_summer), float(bias_T2_winter))
    else:
        bias_T2_step = np.full(nt, float(bias_T2))
    bias_T2_step = bias_T2_step.reshape((nt,) + (1,) * (T2_raw.ndim - 1))
    DATA['T2'] = (DATA.T2.dims, T2_raw + bias_T2_step)
    T2 = DATA.T2.values.copy()

    # T2 bias at constant vapour pressure: RH follows the new saturation pressure
    # (same Sonntag function COSIPY uses to turn RH into vapour pressure); LWin is scaled further below
    if np.any(bias_T2_step != 0.0):
        _es_raw = np.array([method_EW_Sonntag(x) for x in T2_raw])
        _es_new = np.array([method_EW_Sonntag(x) for x in T2])
        DATA['RH2'] = (DATA.RH2.dims, np.minimum(DATA.RH2.values * _es_raw / _es_new, 100.0))
    RH2 = DATA.RH2.values
    PRES = DATA.PRES.values
    G = DATA.G.values
    #U2 = DATA.U2.values

    U2 = DATA.U2.values * mult_factor_WS
    #--------------------------------------------
    # Checks for optional input variables
    #--------------------------------------------
    
    # Snowfall factor per time step = SRF x K_AVA (each one only if in the forcing and switched on in the config).
    # Both are constant within an outline period and change when the next outline starts, so they may vary in
    # time. NaN -> 1 (no redistribution / no avalanche input).
    use_srf = ("SRF" in DATA) and (Config.use_srf is True)
    use_ava = ("K_AVA" in DATA) and (Config.use_ava is True)

    def _per_step(name):
        a = np.atleast_1d(np.asarray(DATA[name].values, dtype="float64"))
        a = np.where(np.isfinite(a), a, 1.0)
        if a.size not in (1, nt):
            raise ValueError(f"{name} has {a.size} values, expected 1 or {nt}")
        return np.broadcast_to(a, (nt,))

    snow_factor = np.ones(nt)
    if use_srf:
        snow_factor = snow_factor * _per_step("SRF")
    if use_ava:
        snow_factor = snow_factor * _per_step("K_AVA")

    rrr_summer_months = (6,7,8,9)          # JJAS
    if (rrr_factor_summer is not None) and (rrr_factor_winter is not None):
        rrr_factor = np.where(np.isin(_months, rrr_summer_months), rrr_factor_summer, rrr_factor_winter)
    else:
        rrr_factor = np.full(nt, mult_factor_RRR, dtype='float64')
    
    if ('SNOWFALL' in DATA) and ('RRR' in DATA):
        SNOWF = DATA.SNOWFALL.values * rrr_factor
        RRR = DATA.RRR.values * rrr_factor
    elif 'SNOWFALL' in DATA:
        SNOWF = DATA.SNOWFALL.values * rrr_factor
        RRR = None
        RAIN = None
    else:
        SNOWF = None
        RRR = DATA.RRR.values * rrr_factor

    # Use RRR rather than snowfall?
    if Config.force_use_TP:
        SNOWF = None

    LWin = np.array(nt * [None])
    N = np.array(nt * [None])
    # LWin: scaled with the T2 bias (Stefan-Boltzmann, clouds/terrain of the forcing kept),
    # then the constant radiative offset bias_LWIN
    if ('LWin' in DATA) and ('N' in DATA):
        DATA['LWin'] = np.clip(DATA['LWin'] * (T2 / T2_raw) ** 4 + bias_LWIN, a_min=0.0, a_max=None)
        LWin = DATA.LWin.values
        N = DATA.N.values
    elif 'LWin' in DATA:
        DATA['LWin'] = np.clip(DATA['LWin'] * (T2 / T2_raw) ** 4 + bias_LWIN, a_min=0.0, a_max=None)
        LWin = DATA.LWin.values
    else:
        LWin = None
        N = DATA.N.values

    # Use N rather than LWin
    if Config.force_use_N:
        LWin = None

    SLOPE = 0.
    if 'SLOPE' in DATA:
        SLOPE = DATA.SLOPE.values
    _elev = float(DATA["HGT"].values) #time constant!

    # Initial snow albedo and surface temperature for Bougamont et al. 2005 albedo
    surface_temperature = 270.0
    albedo_snow = albedo_fresh_snow
    if WRF_X_CSPY:
        albedo_snow = albedo_firn
    # Albedo memory (albedo_memory in constants.toml): [stored snow albedo below the fresh layer, fresh-layer SWE
    # [kg m-2], fresh-layer height [m]]
    albedo_memory_state = [np.nan, 0.0, 0.0] if albedo_memory else None

    if Config.stake_evaluation:
        # Create pandas dataframe for stake evaluation
        _df = pd.DataFrame(index=stake_data.index, columns=['mb','snowheight'], dtype='float')

    # Initial values
    MB_cum = 0
    #Ligtenberg annual accum.
    _years  = pd.DatetimeIndex(DATA.time.values).year.values
    HYDRO_YEAR = np.where(_months < 10,_years, _years+1)
    annual_mass_balances = np.empty(0)
    ACC_cum_year = 0.0
    accumulation = 1 #results in 1000 m w.e. for first year

    #--------------------------------------------
    # TIME LOOP
    #--------------------------------------------
    for t in np.arange(nt):
        
        # Check grid
        GRID.grid_check()

        # get seconds since start
        # timestamp = dt*t
        # if Config.WRF_X_CSPY:
            # timestamp = np.float64(DATA.CURR_SECS.values)

        # Calc fresh snow density
        if densification_method != 'constant':
            density_fresh_snow = np.maximum(109.0+6.0*(T2[t]-273.16)+26.0*np.sqrt(U2[t]), 50.0)
        else:
            density_fresh_snow = constant_density 

        # Derive snowfall [m] and rain rates [m w.e.]
        if (SNOWF is not None) and (RRR is not None):
            SNOWFALL = SNOWF[t]
            RAIN = RRR[t]-SNOWFALL*(density_fresh_snow/water_density) * 1000.0
        elif SNOWF is not None:
            SNOWFALL = SNOWF[t]
        elif RRR is not None:
            if precippartition_method == "Hantel2000":
                # Else convert total precipitation [mm] to snowheight [m]; liquid/solid fraction
                SNOWFALL = (RRR[t]/1000.0)*(water_density/density_fresh_snow)*(0.5*(-np.tanh(((T2[t]-zero_temperature) - center_snow_transfer_function) * spread_snow_transfer_function) + 1.0))
                RAIN = RRR[t]-SNOWFALL*(density_fresh_snow/water_density) * 1000.0
            elif precippartition_method == "Ding2014":
                #Ding, et al. 2014. J. Hydrol http://dx.doi.org/10.1016/j.jhydrol.2014.03.038 also in T&C
                Twb = calc_wetbulb_temperature(T2[t], RH2[t], PRES[t])
                elev = _elev
                offset_T = center_snow_transfer_function
                RAIN, Pr_sno = partition_precipitation (T2[t], Twb, RH2[t], PRES[t], RRR[t], elev, offset_T) 
                SNOWFALL = (Pr_sno / 1000.0) * (water_density/density_fresh_snow)
        else:
            raise ValueError("No SNOWFALL or RRR data provided.")
        
        # if snowfall is smaller than the threshold
        if SNOWFALL<minimum_snowfall:
            SNOWFALL = 0.0

        # Apply SRF x K_AVA (1.0 if neither is used)
        SNOWFALL = SNOWFALL * snow_factor[t]

        # if rainfall is smaller than the threshold
        if RAIN<(minimum_snowfall*(density_fresh_snow/water_density)*1000.0):
            RAIN = 0.0

        if SNOWFALL > 0.0:
            # Add a new snow node on top
            GRID.add_fresh_snow(SNOWFALL, density_fresh_snow, np.minimum(float(T2[t]),zero_temperature), 0.0, dt, int(HYDRO_YEAR[t]))
        else:
            GRID.set_fresh_snow_props_update_time(dt)

        # Guarantee that solar radiation is greater equal zero
        G[t] = max(G[t],0.0)

        #--------------------------------------------
        # Merge grid layers, if necessary
        #--------------------------------------------
        GRID.update_grid()

        #--------------------------------------------
        # Calculate albedo and roughness length changes if first layer is snow
        #--------------------------------------------
        # snowfall mass of this step [kg m-2] for the albedo refresh (SNOWFALL is snow height after SRF x K_AVA)
        fresh_swe = float(SNOWFALL * density_fresh_snow) if SNOWFALL > 0.0 else 0.0
        alpha, albedo_snow = updateAlbedo(GRID,surface_temperature,albedo_snow,opt_dict,fresh_swe,albedo_memory_state)

        #--------------------------------------------
        # Update roughness length
        #--------------------------------------------
        z0 = updateRoughness(GRID,opt_dict)

        #--------------------------------------------
        # Surface Energy Balance
        #--------------------------------------------
        # Calculate net shortwave radiation
        SWnet = G[t] * (1 - alpha)

        # Penetrating SW radiation and subsurface melt
        if SWnet > 0.0:
            subsurface_melt, G_penetrating = penetrating_radiation(GRID, SWnet, dt)
        else:
            subsurface_melt = 0.0
            G_penetrating = 0.0

        # Calculate residual net shortwave radiation (penetrating part removed)
        sw_radiation_net = SWnet - G_penetrating

        if (LWin is not None) and (N is not None):
            # Find new surface temperature
            fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
                 ground_heat_flux, rain_heat_flux, rho, Lv, MOL, Cs_t, Cs_q, q0, q2 \
                 = update_surface_temperature(GRID, dt, z, z0, T2[t], RH2[t], PRES[t], sw_radiation_net, \
                 U2[t], RAIN, SLOPE, LWin=LWin[t], N=N[t])
        elif LWin is not None:
            fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
                ground_heat_flux, rain_heat_flux, rho, Lv, MOL, Cs_t, Cs_q, q0, q2 \
                = update_surface_temperature(GRID, dt, z, z0, T2[t], RH2[t], PRES[t], sw_radiation_net, \
                                             U2[t], RAIN, SLOPE, LWin=LWin[t])
        else:
            # Find new surface temperature (LW is parametrized using cloud fraction)
            fun, surface_temperature, lw_radiation_in, lw_radiation_out, sensible_heat_flux, latent_heat_flux, \
                ground_heat_flux, rain_heat_flux, rho, Lv, MOL, Cs_t, Cs_q, q0, q2 \
                = update_surface_temperature(GRID, dt, z, z0, T2[t], RH2[t], PRES[t], sw_radiation_net, \
                                             U2[t], RAIN, SLOPE, N=N[t])

        #--------------------------------------------
        # Surface mass fluxes [m w.e.q.]
        #--------------------------------------------
        if surface_temperature < zero_temperature:
            sublimation = min(latent_heat_flux / (water_density * lat_heat_sublimation), 0.) * dt
            deposition = max(latent_heat_flux / (water_density * lat_heat_sublimation), 0.) * dt
            evaporation = 0.
            condensation = 0.
        else:
            sublimation = 0.
            deposition = 0.
            evaporation = min(latent_heat_flux / (water_density * lat_heat_vaporize), 0.) * dt
            condensation = max(latent_heat_flux / (water_density * lat_heat_vaporize), 0.) * dt

        #--------------------------------------------
        # Melt process - mass changes of snowpack (melting, sublimation, deposition, evaporation, condensation)
        #--------------------------------------------
        # Melt energy in [W m^-2 or J s^-1 m^-2]
        melt_energy = max(
            0,
            sw_radiation_net
            + lw_radiation_in
            + lw_radiation_out
            + ground_heat_flux
            + rain_heat_flux
            + sensible_heat_flux
            + latent_heat_flux
        )

        # Convert melt energy to m w.e.q.
        melt = melt_energy * dt / (1000 * lat_heat_melting)

        # Remove melt [m w.e.q.]
        lwc_from_melted_layers = GRID.remove_melt_weq(melt - sublimation - deposition)
        if albedo_memory_state is not None and albedo_memory_state[1] > 0.0:
            # surface melt and sublimation remove the fresh layer first [kg m-2]; its height shrinks in proportion
            swe_left = max(albedo_memory_state[1] - 1000.0 * max(melt - sublimation - deposition, 0.0), 0.0)
            albedo_memory_state[2] = albedo_memory_state[2] * swe_left / albedo_memory_state[1]
            albedo_memory_state[1] = swe_left

        #--------------------------------------------
        # Percolation
        #--------------------------------------------
        nn = GRID.number_nodes
        h = GRID.get_height()
        if nn < 2 or len(h) != nn:
            raise RuntimeError(
                f"degenerate grid: number_nodes={nn}, len(heights)={len(h)}, "
                f"total_h={np.sum(h):.2f}, t={t}"
            )

        Q  = percolation(GRID, melt + condensation + RAIN/1000.0 + lwc_from_melted_layers, dt)

        #--------------------------------------------
        # Refreezing
        #--------------------------------------------
        water_refreezed = refreezing(GRID)

        #--------------------------------------------
        # Solve the heat equation
        #--------------------------------------------
        # in cosipy_core.py, just before line 506
        solveHeatEquation(GRID, dt)

        #--------------------------------------------
        # Calculate new density to densification
        #--------------------------------------------
        if (HYDRO_YEAR[t] != HYDRO_YEAR[max(t - 1, 0)]) and (HYDRO_YEAR[t] != (HYDRO_YEAR[0] + 1)):
            annual_mass_balances = np.append(annual_mass_balances, ACC_cum_year)
            accumulation = np.mean(annual_mass_balances)
            ACC_cum_year = 0.0

        densification(GRID, SLOPE, dt, accumulation)

        #--------------------------------------------
        # Calculate mass balance
        #--------------------------------------------
        surface_mass_balance = (
            SNOWFALL * (density_fresh_snow / water_density)
            - melt
            + sublimation
            + deposition
            + evaporation
        )
        internal_mass_balance = water_refreezed - subsurface_melt
        mass_balance = surface_mass_balance + internal_mass_balance



        # ---- Debug: Dichteprofil einmal pro Jahr fuer ein einzelnes Band ----
        # DEBUG_ELEV = None schaltet ab. Toleranz, weil HGT selten exakt trifft.
        #DEBUG_ELEV = None
        if DEBUG_ELEV is not None and abs(_elev - DEBUG_ELEV) < 10.0:
            _ts = DATA.isel(time=t).time.values

            _tav = np.asarray(GRID.get_average_temperature())
            _tt = np.asarray(GRID.get_temperature())

            if (pd.Timestamp(_ts).month == 9) and (pd.Timestamp(_ts).day == 30) \
               and (pd.Timestamp(_ts).hour == 0):
                _rho = np.asarray(GRID.get_density())
                _h = np.asarray(GRID.get_height())
                _d = np.cumsum(_h) - 0.5 * _h
                _n830 = int((_rho >= 830).sum())
                _z830 = _d[_rho >= 830][0] if _n830 > 0 else np.nan
                _n = len(_d)
                _i1 = min(np.searchsorted(_d, 1.0), _n -1)
                _i5 = min(np.searchsorted(_d, 5.0), _n -1)
                _i10 = min(np.searchsorted(_d, 10.0), _n-1)

                print(f"[{pd.Timestamp(_ts).year}] HGT={_elev:.0f} "
                      f"nlay={GRID.get_number_layers():4d} "
                      f"H={_h.sum():6.2f}m  rho: top={_rho[0]:5.0f} "
                      f"1m={_rho[_i1]:5.0f} "
                      f"5m={_rho[_i5]:5.0f} "
                      f"10m={_rho[_i10]:5.0f} "
                      f"max={_rho.max():5.0f}  n>=830: {_n830:3d}  z830={_z830:6.2f}m",
                      flush=True)

                print(f"      T[0]={_tt[0]-273.16:+6.1f} Tavg[0]={_tav[0]-273.16:+6.1f}  "
                      f"T[10]={_tt[min(10,len(_tt)-1)]-273.16:+6.1f} "
                      f"Tavg[10]={_tav[min(10,len(_tav)-1)]-273.16:+6.1f}", 
                      flush=True)

                _mass = float(np.sum(np.asarray(GRID.get_height())
                                     * np.asarray(GRID.get_density())) / 1000.0)
                print(f"        T[0]={_tt[0]-273.16:+6.1f} Tavg[0]={_tav[0]-273.16:+6.1f}  "
                      f"T[10]={_tt[min(10,len(_tt)-1)]-273.16:+6.1f} "
                      f"Tavg[10]={_tav[min(10,len(_tav)-1)]-273.16:+6.1f}  "
                      f"Masse={_mass:9.3f} m w.e.", flush=True)

        # internal_mass_balance2 = melt-Q  + subsurface_melt
        # mass_balance_check = surface_mass_balance + internal_mass_balance2

        # Cumulative mass balance for stake evaluation 
        MB_cum = MB_cum + mass_balance
        ACC_cum_year = ACC_cum_year + SNOWFALL * (density_fresh_snow / water_density) + deposition + sublimation

        # Store cumulative MB in pandas frame for validation
        if stake_names:
            if DATA.isel(time=t).time.values in stake_data.index:
                _df['mb'].loc[DATA.isel(time=t).time.values] = MB_cum 
                _df['snowheight'].loc[DATA.isel(time=t).time.values] = GRID.get_total_snowheight()

        # Save results -- standalone cosipy case
        if not WRF_X_CSPY:
            _RAIN[t] = RAIN
            _SNOWFALL[t] = SNOWFALL * (density_fresh_snow/water_density)
            _LWin[t] = lw_radiation_in
            _LWout[t] = lw_radiation_out
            _H[t] = sensible_heat_flux
            _LE[t] = latent_heat_flux
            _B[t] = ground_heat_flux
            _QRR[t] = rain_heat_flux
            _MB[t] = mass_balance
            _surfMB[t] = surface_mass_balance
            _Q[t] = Q
            _SNOWHEIGHT[t] = GRID.get_total_snowheight()
            _hy = np.asarray(GRID.get_hydro_year())
            _lh = np.asarray(GRID.get_height())
            _rho = np.asarray(GRID.get_density())
            _CY_SNOWHEIGHT[t] = float(np.sum(_lh[(_hy == HYDRO_YEAR[t]) & (_rho < Constants.snow_ice_threshold)]))
            _lwc = np.asarray(GRID.get_liquid_water_content())
            _LWC_00_10[t] = zone_max_lwc(_lh, _lwc, *LWC_ZONES["LWC_00_10"])
            _LWC_10_50[t] = zone_max_lwc(_lh, _lwc, *LWC_ZONES["LWC_10_50"])
            _LWC_DEEP[t] = zone_max_lwc(_lh, _lwc, *LWC_ZONES["LWC_DEEP"])

            _TOTALHEIGHT[t] = GRID.get_total_height()
            _TS[t] = surface_temperature
            _ALBEDO[t] = alpha
            _NLAYERS[t] = GRID.get_number_layers()
            _ME[t] = melt_energy
            _intMB[t] = internal_mass_balance
            _EVAPORATION[t] = evaporation
            _SUBLIMATION[t] = sublimation
            _CONDENSATION[t] = condensation
            _DEPOSITION[t] = deposition
            _REFREEZE[t] = water_refreezed
            _subM[t] = subsurface_melt
            _Z0[t] = z0
            _surfM[t] = melt
            _MOL[t] = MOL
            _new_snow_height[t], _new_snow_timestamp[t], _old_snow_timestamp[t] = GRID.get_fresh_snow_props()

            if Config.full_field:
                if GRID.get_number_layers()>max_layers:
                    raise ValueError('Maximum number of layers reached')
                _LAYER_HEIGHT[t, 0:GRID.get_number_layers()] = GRID.get_height()
                _LAYER_RHO[t, 0:GRID.get_number_layers()] = GRID.get_density()
                _LAYER_T[t, 0:GRID.get_number_layers()] = GRID.get_temperature()
                _LAYER_LWC[t, 0:GRID.get_number_layers()] = GRID.get_liquid_water_content()
                _LAYER_CC[t, 0:GRID.get_number_layers()] = GRID.get_cold_content()
                _LAYER_POROSITY[t, 0:GRID.get_number_layers()] = GRID.get_porosity()
                _LAYER_ICE_FRACTION[t, 0:GRID.get_number_layers()] = GRID.get_ice_fraction()
                _LAYER_IRREDUCIBLE_WATER[t, 0:GRID.get_number_layers()] = GRID.get_irreducible_water_content()
                _LAYER_REFREEZE[t, 0:GRID.get_number_layers()] = GRID.get_refreeze()
                _LAYER_TAVG[t, 0:GRID.get_number_layers()] = GRID.get_average_temperature()
                _LAYER_HY[t, 0:GRID.get_number_layers()] = GRID.get_hydro_year()
            else:
                _LAYER_HEIGHT = None
                _LAYER_RHO = None
                _LAYER_T = None
                _LAYER_LWC = None
                _LAYER_CC = None
                _LAYER_POROSITY = None
                _LAYER_ICE_FRACTION = None
                _LAYER_IRREDUCIBLE_WATER = None
                _LAYER_REFREEZE = None
                _LAYER_TAVG = None
                _LAYER_HY = None

        # Save results -- WRF_X_CSPY case
        else:
            _SNOWHEIGHT = GRID.get_total_snowheight()
            _TOTALHEIGHT = GRID.get_total_height()
            _NLAYERS = GRID.get_number_layers()
            _new_snow_height, _new_snow_timestamp, _old_snow_timestamp = GRID.get_fresh_snow_props()

            _LAYER_HEIGHT = np.array(max_layers * [np.nan])
            _LAYER_RHO = np.array(max_layers * [np.nan])
            _LAYER_T = np.array(max_layers * [np.nan])
            _LAYER_LWC = np.array(max_layers * [np.nan])
            _LAYER_ICE_FRACTION = np.array(max_layers * [np.nan])
            if GRID.get_number_layers()>max_layers:
                raise ValueError('Maximum number of layers reached')
            _LAYER_HEIGHT[0:GRID.get_number_layers()] = GRID.get_height()
            _LAYER_RHO[0:GRID.get_number_layers()] = GRID.get_density()
            _LAYER_T[0:GRID.get_number_layers()] = GRID.get_temperature()
            _LAYER_LWC[0:GRID.get_number_layers()] = GRID.get_liquid_water_content()
            _LAYER_ICE_FRACTION[0:GRID.get_number_layers()] = GRID.get_ice_fraction()

            return (None,None,None,None,None,None,lw_radiation_out,sensible_heat_flux,latent_heat_flux, \
                    ground_heat_flux,rain_heat_flux,mass_balance,surface_mass_balance,Q,_SNOWHEIGHT, \
                    _TOTALHEIGHT,surface_temperature,alpha,_NLAYERS,melt_energy,internal_mass_balance, \
                    evaporation,sublimation,condensation,deposition,water_refreezed,subsurface_melt,z0, \
                    melt,_new_snow_height,_new_snow_timestamp,_old_snow_timestamp,MOL,_LAYER_HEIGHT, \
                    _LAYER_RHO,_LAYER_T,_LAYER_LWC,None,None,_LAYER_ICE_FRACTION,None,None,None,None,None)

    if not WRF_X_CSPY:
        if Config.stake_evaluation:
            # Evaluate stakes
            _stat = evaluate(stake_names, stake_data, _df)


        else:
            _stat = None
            _df = None

        # Restart
        RESTART.NLAYERS.values[:] = GRID.get_number_layers()
        RESTART.NEWSNOWHEIGHT.values[:] = _new_snow_height[t]
        RESTART.NEWSNOWTIMESTAMP.values[:] = _new_snow_timestamp[t]
        RESTART.OLDSNOWTIMESTAMP.values[:] = _old_snow_timestamp[t]
        RESTART.LAYER_HEIGHT[0:GRID.get_number_layers()] = GRID.get_height()
        RESTART.LAYER_RHO[0:GRID.get_number_layers()] = GRID.get_density()
        RESTART.LAYER_T[0:GRID.get_number_layers()] = GRID.get_temperature()
        RESTART.LAYER_LWC[0:GRID.get_number_layers()] = GRID.get_liquid_water_content()
        RESTART.LAYER_IF[0:GRID.get_number_layers()] = GRID.get_ice_fraction()
        RESTART.LAYER_TAVG[0:GRID.get_number_layers()] = GRID.get_average_temperature()
        RESTART.LAYER_HY[0:GRID.get_number_layers()] = GRID.get_hydro_year()
        RESTART.BASE_ELEVATION.values[:] = GRID.get_base_elevation()

        # Forcing as COSIPY used it (after bias_T2 incl. the RH2 adjustment, the wind factor and the precipitation
        # factors; RRR before the snowfall threshold and SRF / K_AVA). Written to the output instead of the raw
        # forcing copied from the input file (the raw fields stay available in the forcing file).
        forcing_out = {"T2": np.asarray(T2, dtype=np.float64), "RH2": np.asarray(RH2, dtype=np.float64),
                       "U2": np.asarray(U2, dtype=np.float64)}
        if RRR is not None:
            forcing_out["RRR"] = np.asarray(RRR, dtype=np.float64)

        return (indY,indX,RESTART,_RAIN,_SNOWFALL,_LWin,_LWout,_H,_LE,_B,_QRR, \
            _MB,_surfMB,_Q,_SNOWHEIGHT,_TOTALHEIGHT,_TS,_ALBEDO,_NLAYERS, \
            _ME,_intMB,_EVAPORATION,_SUBLIMATION,_CONDENSATION,_DEPOSITION,_REFREEZE, \
            _subM,_Z0,_surfM,_new_snow_height,_new_snow_timestamp,_old_snow_timestamp,_MOL, \
            _LAYER_HEIGHT,_LAYER_RHO,_LAYER_T,_LAYER_LWC,_LAYER_CC,_LAYER_POROSITY,_LAYER_ICE_FRACTION, \
            _LAYER_IRREDUCIBLE_WATER,_LAYER_REFREEZE,_LAYER_TAVG,_LAYER_HY,_CY_SNOWHEIGHT,_LWC_00_10,_LWC_10_50,_LWC_DEEP,stake_names,_stat,_df,forcing_out)
