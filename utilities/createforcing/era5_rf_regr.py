import sys
import pandas as pd
import numpy as np
import salem
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
sys.path.append('../../')

#### Load functions ####
def datetime_index(df, time_var, rename, old_time_name):
    if rename:
        df[time_var] = pd.to_datetime(df.pop(old_time_name))
    else:
        df[time_var] = pd.to_datetime(df[time_var])
    df.set_index(time_var, inplace=True)
    return df

### Load Path and Data ###
era_path = "/data/scratch/richteny/thesis/cosipy_test_space/utilities/createforcing/"
aws_path = "/data/scratch/richteny/thesis/AWS/Abramov/"
glacier_outline = "../../data/static/Shapefiles/abramov_rgi6.shp"
dem = "/data/projects/topoclif/input-data/DEMs/HMA_alos-jaxa.tif"

output_path = aws_path+"RF/"
os.makedirs(output_path, exist_ok=True)

#Helper Functions #
#Heavily influenced by https://gis.stackexchange.com/questions/260304/extract-raster-values-within-shapefile-with-pygeoprocessing-or-gdal
def get_glacier_elev(dem):
    gla_shp = salem.read_shapefile(glacier_outline)
    geoms = gla_shp.geometry.values
    geometry = geoms[0]
    geoms = [mapping(geoms[0])]
    #this assumes there is one continuous polygon
    with rasterio.open(dem) as src:
        out_image, out_transform = mask(src, geoms, crop=True)
    no_data=src.nodata
    data = out_image[0,:,:]
    elev = np.extract(data != no_data, data)
    elev = elev[elev != no_data]
    print(elev)
    mean_alt = np.nanmean(elev)
    return mean_alt

mean_glacier_alt = get_glacier_elev(dem)
print(mean_glacier_alt)

#from rasterstats import zonal_stats
#stats = zonal_stats(glacier_outline, dem)
#print(stats[0])

aws_df = pd.read_csv(aws_path+"Abramov_AWS_2014-01-01_2020-01-01.csv", index_col=False)
aws_df = datetime_index(aws_df, 'time', True, 'index')
aws_df.drop('Unnamed: 0', axis=1, inplace=True)

era5_df = pd.read_csv(era_path+"Abramov_ERA5_1999_2021.csv", index_col=False)
era5_df = datetime_index(era5_df, 'time', True, 'TIMESTAMP')
era5_df.drop('Unnamed: 0', axis=1, inplace=True)

start_date = "2010-01-01"
end_date = "2020-01-01"
era5_df_full = era5_df.loc[start_date:end_date]
#print(era5_df_full)

era5_df = era5_df[era5_df.index.isin(aws_df.index)]

aws_height = 4102 #Kronenberg 2022 says 4100
cosipy_height = mean_glacier_alt

print(aws_df.isnull().sum())
print(era5_df.isnull().sum())

print(np.nanmax(aws_df['T2']))
print(np.nanmin(aws_df['T2']))
#Either drop NaNs, or interpolate values
aws_df = aws_df.interpolate()
#aws_df.dropna(inplace=True)
#era5_df = era5_df[era5_df.index.isin(aws_df)]
print(aws_df)
print(era5_df)
print(aws_df.isnull().sum())
print(era5_df.isnull().sum())

#Some diagnstocis
print(np.nanmax(aws_df['T2']))
print(np.nanmin(aws_df['T2']))
print(np.nanmax(era5_df['T2']-273.15))
print(np.nanmin(era5_df['T2']-273.15))


def do_rf_prediction(var_to_predict, ev_thres, n_runs):

    print("Starting random forest regression for var:", var_to_predict)
    #era5_df.drop(['U2'], axis=1, inplace=True)
    if var_to_predict == 'T2':
        aws_t2m = pd.DataFrame(aws_df[var_to_predict]+273.15)
    else:
        aws_t2m = pd.DataFrame(aws_df[var_to_predict])
    aws_t2m.rename(columns={var_to_predict: '{}_interp'.format(var_to_predict)}, inplace=True)

    era5_rf = era5_df.copy()
    era5_rf['hour'] = era5_rf.index.hour
    era5_rf['month'] = era5_rf.index.month
    #era5_rf['year'] = era5_rf.index.year

    #Prepare for Random Forest Model#
    rf_df = pd.DataFrame(aws_t2m['{}_interp'.format(var_to_predict)]).join(era5_rf)
    print(rf_df.isnull().sum())

    rf_df.reset_index(inplace=True)
    rf_df_label = rf_df['time']
    rf_df.drop('time', axis=1, inplace=True)

    era5_rf_full = era5_df_full.copy()
    era5_rf_full['hour'] = era5_rf_full.index.hour
    era5_rf_full['month'] = era5_rf_full.index.month
    #era5_rf['year'] = era5_rf.index.year
    era5_rf_full.reset_index(inplace=True)
    era5_rf_label = era5_rf_full['time']
    era5_rf_full.drop('time', axis=1, inplace=True)

    ev_thres = ev_thres # Min. explained variance to consider RF model for ensemble result
    n_ensemble_runs = n_runs  # Number of times to perform the GridSearch
    #from trial and error values below
    parameters = {'max_depth': [10, 100, 200, None], 'n_estimators':[10, 50, 100, 200 ],
                  'min_samples_leaf': [2], 'min_samples_split': [2]} 
    
    #Check predictors fi_df
    #select predictors here

    df_predict = rf_df.copy()
    df_predict.drop(['{}_interp'.format(var_to_predict)], axis=1, inplace=True)
    
    X = rf_df.drop(['{}_interp'.format(var_to_predict)], axis=1)
    y = rf_df['{}_interp'.format(var_to_predict)]

    # Timeseries DF for prediction to determine accuraccy
    df_predict_timeseries = pd.DataFrame()
    df_predict_timeseries['date'] = pd.to_datetime(rf_df_label)
    df_predict_timeseries.set_index('date', inplace=True)

    df_ensemble_timeseries = df_predict_timeseries.copy()

    #Repeat timeseries DF for full ERA5 timeframe
    df_predict_full_timeseries = pd.DataFrame()
    df_predict_full_timeseries['date'] = pd.to_datetime(era5_rf_label)
    df_predict_full_timeseries.set_index('date', inplace=True)

    df_ensemble_full_timeseries = df_predict_full_timeseries.copy()

    # Set up EV scores
    ev_scores = []
    best_ev_score = 0
    best_model = None

    best_param_df = pd.DataFrame(columns={'Best_param_set': [],
                                          'Explained_Variance': [],
                                          'RMSE': []})

    fi_df = pd.DataFrame(columns=['EV'] + list(X.columns))

    for ens_run in range(1, n_ensemble_runs + 1):

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
    
        rfr = RandomForestRegressor()                        
        regr = GridSearchCV(rfr, parameters, n_jobs=8, cv=5)
        regr.fit(train_X, train_y)
    
        ev_score = explained_variance_score(test_y, regr.predict(test_X))
        ev_scores.append(ev_score)
        print('    Explained variance: '+ str(ev_score))

        if best_ev_score < ev_score:
            best_ev_score = ev_score
            best_model = regr

        # model = RandomForestRegressor(**regr.best_params_, n_jobs=-1)

        # Feature Importances
        fi = regr.best_estimator_.feature_importances_        
        fi_row = dict(zip(['EV'] + list(X.columns), [ev_score] + list(fi)))
        fi_df = fi_df.append(fi_row, ignore_index=True)       

        # print('\n    Feature importances: '+ str(regr.best_estimator_.feature_importances_))

        df_predict_timeseries['{}_pred_'.format(var_to_predict)+ str(ens_run)] = regr.predict(df_predict)
    
        #Full TS
        df_predict_full_timeseries['{}_pred_'.format(var_to_predict)+ str(ens_run)] = regr.predict(era5_rf_full)

        r2 = r2_score(test_y, regr.predict(test_X))

        best_param_df = best_param_df.append({'Best_param_set': str(regr.best_params_),
                                              'Explained_Variance': str(explained_variance_score(test_y, regr.predict(test_X))),
                                              'RMSE': str(mean_squared_error(test_y, regr.predict(test_X), squared=False)),
                                              'r2': str(r2)}, ignore_index=True)
        
    # Generate ensemble best-of prediction    
    runs_above_ev_thres = []
    del_list = []
    for i in range(1, len(ev_scores) + 1):
        if ev_scores[i-1] > ev_thres:           
            df_ensemble_timeseries['{}_pred_'.format(var_to_predict)+ str(i)] = df_predict_timeseries['{}_pred_'.format(var_to_predict)+ str(i)] 
            df_ensemble_timeseries['{}_pred_'.format(var_to_predict)+ str(i) +'_wgtd'] = df_ensemble_timeseries['{}_pred_'.format(var_to_predict)+ str(i)] * ev_scores[i-1]
            #Now for full timeseries
            df_ensemble_full_timeseries['{}_pred_'.format(var_to_predict)+ str(i)] = df_predict_full_timeseries['{}_pred_'.format(var_to_predict)+ str(i)] 
            df_ensemble_full_timeseries['{}_pred_'.format(var_to_predict)+ str(i) +'_wgtd'] = df_ensemble_full_timeseries['{}_pred_'.format(var_to_predict)+ str(i)] * ev_scores[i-1]

            runs_above_ev_thres.append(i)
        else:
            del_list.append(i-1)

    # Generate weighted mean
    ev_scores_ensemble = [ i for i in ev_scores if ev_scores.index(i) not in del_list ]
    ev_ensemble_sum = np.array(ev_scores_ensemble).sum()
    df_ensemble_timeseries['{}_pred_ensemble'.format(var_to_predict)] = 0
    df_ensemble_full_timeseries['{}_pred_ensemble'.format(var_to_predict)] = 0

    for run in runs_above_ev_thres:
        df_ensemble_timeseries['{}_pred_ensemble'.format(var_to_predict)] = df_ensemble_timeseries['{}_pred_ensemble'.format(var_to_predict)] + df_ensemble_timeseries['{}_pred_'.format(var_to_predict)+ str(run) +'_wgtd']
        df_ensemble_full_timeseries['{}_pred_ensemble'.format(var_to_predict)] = df_ensemble_full_timeseries['{}_pred_ensemble'.format(var_to_predict)] + df_ensemble_full_timeseries['{}_pred_'.format(var_to_predict)+ str(run) +'_wgtd']
    
    df_ensemble_timeseries['{}_pred_ensemble'.format(var_to_predict)] = df_ensemble_timeseries['{}_pred_ensemble'.format(var_to_predict)] / ev_ensemble_sum
    df_ensemble_full_timeseries['{}_pred_ensemble'.format(var_to_predict)] = df_ensemble_full_timeseries['{}_pred_ensemble'.format(var_to_predict)] / ev_ensemble_sum

    df_ensemble_timeseries = df_ensemble_timeseries[df_ensemble_timeseries.columns.drop(list(df_ensemble_timeseries.filter(regex='wgtd')))]
    df_ensemble_full_timeseries = df_ensemble_full_timeseries[df_ensemble_full_timeseries.columns.drop(list(df_ensemble_full_timeseries.filter(regex='wgtd')))]

    # Export run characteristics (EVs, MSEs)
    fi_df.to_hdf(output_path + '{}_feature-importances.h5'.format(var_to_predict), key='EV', format='table')

    # Export best-of prediction
    df_ensemble_timeseries.to_hdf(output_path + '{}_predict-timeseries.h5'.format(var_to_predict), key='{}_pred_ensemble'.format(var_to_predict), format='table')
    df_ensemble_full_timeseries.to_hdf(output_path + '{}_predict-full_timeseries.h5'.format(var_to_predict), key='{}_pred_ensemble'.format(var_to_predict), format='table')

    best_param_df.to_hdf(output_path + '{}_best_param_df.h5'.format(var_to_predict), key="Best_param", format="table")

    df_ensemble_timeseries['{}_interp'.format(var_to_predict)] = rf_df['{}_interp'.format(var_to_predict)].values

    # Export best model    
    with open(output_path + '{}_best-model.pkl'.format(var_to_predict), 'wb') as model_file:
        pickle.dump(best_model, model_file)  

    label_dic = {'T2': 'Air temperature at 2m (°C)',
                 'U2': 'Wind speed at 2m (m/s)',
                 'PRES': 'Air pressure (hPa)'}

    my_dpi = 300
    fig = plt.figure(figsize=(26,12), dpi=my_dpi)
    ax= fig.add_subplot(111) 
    ax.plot(df_ensemble_timeseries['{}_interp'.format(var_to_predict)], color="darkslategrey", label= 'AWS T2M interpolated', zorder=4)
    ax.plot(df_ensemble_timeseries['{}_pred_ensemble'.format(var_to_predict)], "-g", label= "Ensemble pred", zorder=6)
    ax.legend(loc='best')
    ax.set_xlabel('Date')
    ax.set_ylabel(label_dic[var_to_predict])
    plt.savefig(output_path + '{}_first_try_rf_predict_300dpi'.format(var_to_predict), bbox_inches="tight")

    # Plot full timeseries
    fig = plt.figure(figsize=(26,12), dpi=my_dpi)
    ax= fig.add_subplot(111) 
    ax.plot(df_ensemble_timeseries['{}_interp'.format(var_to_predict)],  color="darkslategrey", label= 'AWS T2M interpolated', zorder=4)
    ax.plot(df_ensemble_full_timeseries['{}_pred_ensemble'.format(var_to_predict)], "-g", label= "Ensemble pred", zorder=6)
    ax.legend(loc='best')
    ax.set_xlabel('Date')
    ax.set_ylabel(label_dic[var_to_predict])
    plt.savefig(output_path + '{}_first_try_rf_full_predict_300dpi'.format(var_to_predict), bbox_inches="tight")


for var in ['RH2','T2','U2','PRES']:
    if var == 'U2':
        do_rf_prediction(var_to_predict=var, ev_thres=0.469, n_runs=30)
    elif var == 'T2':
        do_rf_prediction(var_to_predict=var, ev_thres=0.964, n_runs=30)
    elif var == 'RH2':
        do_rf_prediction(var_to_predict=var, ev_thres=0.727, n_runs=30)
    else:
        do_rf_prediction(var_to_predict=var, ev_thres=0.984, n_runs=30)

print("Performed Random Forest Regression. Values are calculated at AWS elevation.")

