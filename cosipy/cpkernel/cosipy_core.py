import numpy as np
import pandas as pd

from cosipy.config import Config
from cosipy.constants import Constants
from cosipy.cpkernel.init import init_snowpack, load_snowpack
from cosipy.cpkernel.io import IOClass
from cosipy.modules.albedo import updateAlbedo
from cosipy.modules.densification import densification
from cosipy.modules.evaluation import evaluate, resample_output, create_tsl_df, eval_tsl, resample_by_hand
from cosipy.modules.heatEquation import solveHeatEquation
from cosipy.modules.penetratingRadiation import penetrating_radiation
from cosipy.modules.percolation import percolation
from cosipy.modules.refreezing import refreezing
from cosipy.modules.roughness import updateRoughness
from cosipy.modules.surfaceTemperature import update_surface_temperature
from cosipy.modules.surfaceTemperature import method_EW_Sonntag


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
    gam = cp * 100.0 * Pre_Pa / (0.622 * lheat)

    esat_Pa = esat * 100.0
    ea_Pa = ea * 100.0
    del_ = (4098.0 * esat_Pa) / ((237.3 + Ta)**2)

    Twb = Ta - (esat_Pa - ea_Pa) / (gam + del_)
    return Twb

def partition_precipitation(TK: float, Twb: float, RH: float, pressure: float, RRR: float) -> tuple:
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
    Zref = -((Ta + 15.0)/2 + 273.15) * (Rd / g) * np.log(pressure / P_Ref)
    Zref = Zref / 1000.0

    # relative humidity fraction
    rh_frac = RH / 100.0

    # threshold parameters
    dT_ = 0.215 - 0.099 * rh_frac + 1.018 * rh_frac**2
    dS_ = 2.374 - 1.634 * rh_frac
    T0_ = -5.87 - 0.1042 * Zref + 0.0885 * Zref**2 + 16.06 * rh_frac - 9.614 * rh_frac**2

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

    # Replace values from constants.py if coupled
    # TODO: This only affects the current module scope instead of global.
    if WRF_X_CSPY:
        dt = int(DATA.DT.values)
        max_layers = int(DATA.max_layers.values)
        z = float(DATA.ZLVL.values)

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

        _LAYER_HEIGHT = init_nan_array_2d(nt, max_layers)
        _LAYER_RHO = init_nan_array_2d(nt, max_layers)
        _LAYER_T = init_nan_array_2d(nt, max_layers)
        _LAYER_LWC = init_nan_array_2d(nt, max_layers)
        _LAYER_CC = init_nan_array_2d(nt, max_layers)
        _LAYER_POROSITY = init_nan_array_2d(nt, max_layers)
        _LAYER_ICE_FRACTION = init_nan_array_2d(nt, max_layers)
        _LAYER_IRREDUCIBLE_WATER = init_nan_array_2d(nt, max_layers)
        _LAYER_REFREEZE = init_nan_array_2d(nt, max_layers)


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
    T2 = DATA.T2.values
    RH2 = DATA.RH2.values
    PRES = DATA.PRES.values
    G = DATA.G.values
    U2 = DATA.U2.values

    #--------------------------------------------
    # Checks for optional input variables
    #--------------------------------------------
    if ('SNOWFALL' in DATA) and ('RRR' in DATA):
        SNOWF = DATA.SNOWFALL.values * mult_factor_RRR
        RRR = DATA.RRR.values * mult_factor_RRR
    elif 'SNOWFALL' in DATA:
        SNOWF = DATA.SNOWFALL.values * mult_factor_RRR
        RRR = None
        RAIN = None
    else:
        SNOWF = None
        RRR = DATA.RRR.values * mult_factor_RRR

    # Use RRR rather than snowfall?
    if Config.force_use_TP:
        SNOWF = None

    LWin = np.array(nt * [None])
    N = np.array(nt * [None])
    if ('LWin' in DATA) and ('N' in DATA):
        LWin = DATA.LWin.values
        N = DATA.N.values
    elif 'LWin' in DATA:
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

    # Initial cumulative mass balance variable
    MB_cum = 0

    # Initial snow albedo and surface temperature for Bougamont et al. 2005 albedo
    surface_temperature = 270.0
    albedo_snow = albedo_fresh_snow
    if WRF_X_CSPY:
        albedo_snow = albedo_firn

    if Config.stake_evaluation:
        # Create pandas dataframe for stake evaluation
        _df = pd.DataFrame(index=stake_data.index, columns=['mb','snowheight'], dtype='float')

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
                RAIN, Pr_sno = partition_precipitation (T2[t], Twb, RH2[t], PRES[t], RRR[t]) 
                SNOWFALL = (Pr_sno / 1000.0) * (water_density/density_fresh_snow)
        else:
            raise ValueError("No SNOWFALL or RRR data provided.")

        # if snowfall is smaller than the threshold
        if SNOWFALL<minimum_snowfall:
            SNOWFALL = 0.0

        # if rainfall is smaller than the threshold
        if RAIN<(minimum_snowfall*(density_fresh_snow/water_density)*1000.0):
            RAIN = 0.0

        if SNOWFALL > 0.0:
            # Add a new snow node on top
            GRID.add_fresh_snow(SNOWFALL, density_fresh_snow, np.minimum(float(T2[t]),zero_temperature), 0.0, dt)
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
        alpha, albedo_snow = updateAlbedo(GRID,surface_temperature,albedo_snow,opt_dict)

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

        #--------------------------------------------
        # Percolation
        #--------------------------------------------
        Q  = percolation(GRID, melt + condensation + RAIN/1000.0 + lwc_from_melted_layers, dt)

        #--------------------------------------------
        # Refreezing
        #--------------------------------------------
        water_refreezed = refreezing(GRID)

        #--------------------------------------------
        # Solve the heat equation
        #--------------------------------------------
        solveHeatEquation(GRID, dt)

        #--------------------------------------------
        # Calculate new density to densification
        #--------------------------------------------
        densification(GRID, SLOPE, dt)

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

        # internal_mass_balance2 = melt-Q  + subsurface_melt
        # mass_balance_check = surface_mass_balance + internal_mass_balance2

        # Cumulative mass balance for stake evaluation 
        MB_cum = MB_cum + mass_balance

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

        return (indY,indX,RESTART,_RAIN,_SNOWFALL,_LWin,_LWout,_H,_LE,_B,_QRR, \
            _MB,_surfMB,_Q,_SNOWHEIGHT,_TOTALHEIGHT,_TS,_ALBEDO,_NLAYERS, \
            _ME,_intMB,_EVAPORATION,_SUBLIMATION,_CONDENSATION,_DEPOSITION,_REFREEZE, \
            _subM,_Z0,_surfM,_new_snow_height,_new_snow_timestamp,_old_snow_timestamp,_MOL, \
            _LAYER_HEIGHT,_LAYER_RHO,_LAYER_T,_LAYER_LWC,_LAYER_CC,_LAYER_POROSITY,_LAYER_ICE_FRACTION, \
            _LAYER_IRREDUCIBLE_WATER,_LAYER_REFREEZE,stake_names,_stat,_df)
