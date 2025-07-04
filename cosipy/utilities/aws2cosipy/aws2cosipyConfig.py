"""
 This is the configuration (init) file for the utility aws2cosipy.
 Please make your changes here.
"""
createLUT = False
#------------------------
# Declare variable names 
#------------------------

# Pressure
PRES_var = 'PRES'

# Temperature
T2_var = 'T2'
in_K = True

# Relative humidity
RH2_var = 'RH2'

# Incoming shortwave radiation
G_var = 'G'

# Precipitation
RRR_var = 'RRR'

# Wind velocity
U2_var = 'U2'

# Incoming longwave radiation
LWin_var = 'LWin'

# Snowfall
SNOWFALL_var = 'SNOWFALL'

# Cloud cover fraction
N_var = 'N'

#------------------------
# Aggregation to hourly data
#------------------------
aggregate = False
aggregation_step = 'H'
ELEV_model = False

# Delimiter in csv file
delimiter = ','

# WRF non uniform grid
WRF = False

#------------------------
# Radiation module 
#------------------------
radiationModule = 'Horayzon2022' # 'Moelg2009', 'Wohlfahrt2016', 'Horayzon2022', 'none'
LUT = False                   # If there is already a Look-up-table for topographic shading and sky-view-factor built for this area, set to True

dtstep = 3600*3               # time step (s)
stationLat = 46.797478          # Latitude of station
tcart = 4.24                    # aka Timezone_Lon - lon of station (~71.586) Station time correction in hour angle units (1 is 4 min)
timezone_lon = 15.0	      # Longitude of station
#station or timezone?

# Zenit threshold (>threshold == zenit): maximum potential solar zenith angle during the whole year, specific for each location
zeni_thld = 89.0              # If you do not know the exact value for your location, set value to 89.0

#------------------------
# Point model 
#------------------------
point_model = True
plon = 46.790453 #90.64
plat = 10.747121 #30.47
hgt = 3048.0 #5665.0

#------------------------
# Interpolation arguments 
#------------------------
stationName = 'HEF'
stationAlt = 3030.0283 #4193.866 #

lapse_T         =  -0.0065    # Temp K per  m Barandun et al. 2018 -0.0048
lapse_RH        =  0.000    # RH % per  m (0 to 1)
lapse_RRR       =  0.000   # mm per m # 0.00064 from Barandun et al. 2018
lapse_SNOWFALL  =  0.0000   # Snowfall % per m (0 to 1)
