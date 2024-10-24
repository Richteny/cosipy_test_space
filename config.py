"""
 This is the COSIPY configuration (init) file.
 Please make your changes here.
"""


#-----------------------------------
# SIMULATION PERIOD 
#-----------------------------------
# Abramov
time_start = "1999-01-01T00:00"
time_end   = "2010-01-01T00:00" 

# Hintereisferner
#time_start = '2018-09-17T08:00'
#time_end   = '2019-07-03T13:00'

#-----------------------------------
# FILENAMES AND PATHS 
#-----------------------------------

time_start_str=(time_start[0:10]).replace('-','')
time_end_str=(time_end[0:10]).replace('-','')

data_path = './data/'

input_netcdf= 'HEF/HEF_COSMO_30m_1999_2010.nc' #Abramov_300m_ERA5mod_spinup_Wohlfahrt_2009-2020.nc' #"Abramov/Abramov_300m_ERA5mod_spinup_Wohlfahrt_2009-2020_crop.nc" #"Abramov/Abramov_300m_ERA5_OldRad_1999_2021.nc" 

output_netcdf = 'HEF_COSMO_30m_1999_2010_HORAYZON_'+time_start_str+'-'+time_end_str+'.nc'



#-----------------------------------
# RESTART 
#-----------------------------------
restart = False                                             # set to true if you want to start from restart file

#-----------------------------------
# Concatenate restarted outputs
#-----------------------------------
merge = False                                              # set to true if you want to concatenate outputs
time_start_old_file = "19821001"                           # requires string of start time from previous file

#-----------------------------------
# STAKE DATA 
#-----------------------------------
stake_evaluation = False 
stakes_loc_file = './data/input/HEF/loc_stakes.csv'         # path to stake location file
stakes_data_file = './data/input/HEF/data_stakes_hef.csv'   # path to stake data file
eval_method = 'rmse'                                        # how to evaluate the simulations ('rmse')
obs_type = 'snowheight'                                     # What kind of stake data is used 'mb' or 'snowheight'

#-----------------------------------
# TRANSIENT SNOWLINE DATA
#-----------------------------------
tsl_evaluation = True 
write_csv_status = True
time_col_obs = 'LS_DATE'
#tsla_col_obs = 'SC_median'
tsla_col_obs = 'TSL_normalized'                                # SC_median for non-normalized and TSL_normalized for normalized
min_snowheight = 0.001                                         # Minimum snowheight in m
tsl_method='conservative'                                      # Possible options are mantra, conservative, grid_search, bare in mind that conservative algorithm assumes there is a spatial consistency in snow-cover
tsl_normalize=True
#tsl_data_file = './data/input/Abramov/snowlines/TSLA_Abramov_filtered_jaso.csv' # path to transient snow line altitudes dataset
tsl_data_file = './data/input/HEF/snowlines/HEF-snowlines-1999-2010_manual.csv'
#-----------------------------------
# Run mutliple Lapse Rates
#-----------------------------------
#lapse_rate_config = True                                     #switch off to run COSIPY without online lapse rate calculation
#lapse_T_range = -0.0061
#lapse_RRR_range = 0.0009
station_altitude = 3030.0283 #4193.866


#-----------------------------------
# STANDARD LAT/LON or WRF INPUT 
#-----------------------------------
# Dimensions
WRF = False                                                 # Set to True if you use WRF as input

northing = 'lat'	                                    # name of dimension	in in- and -output
easting = 'lon'					                        # name of dimension in in- and -output
if WRF:
    northing = 'south_north'                                # name of dimension in WRF in- and output
    easting = 'west_east'                                   # name of dimension in WRF in- and output

# Interactive simulation with WRF
WRF_X_CSPY = False

#-----------------------------------
# COMPRESSION of output netCDF
#-----------------------------------
compression_level = 2                                       # Choose value between 1 and 9 (highest compression)
                                                            # Recommendation: choose 1, 2 or 3 (higher not worthwhile, because of needed time for writing output)
#-----------------------------------
# PARALLELIZATION 
#-----------------------------------
slurm_use = True                                          # use SLURM
workers = None                                              # number of workers, if local cluster is used
local_port = 8786                                           # port for local cluster

#-----------------------------------
# WRITE FULL FIELDS 
#-----------------------------------    
full_field = False                                          # write full fields (2D data) to file
if WRF_X_CSPY:
    full_field = True
    
#-----------------------------------
# TOTAL PRECIPITATION  
#-----------------------------------
force_use_TP = True                                        # If total precipitation and snowfall in input data;
                                                            # use total precipitation

#-----------------------------------
# CLOUD COVER FRACTION  
#-----------------------------------
force_use_N = False                                         # If cloud cover fraction and incoming longwave radiation
                                                            # in input data use cloud cover fraction

#-----------------------------------
# SUBSET  (provide pixel values) 
#-----------------------------------
tile = False
xstart = 20
xend = 40
ystart = 20
yend = 40


