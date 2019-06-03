import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import copy
import time
import itertools
import scipy
import collections
import os
np.random.seed(12345)
maximum_k=7
number_of_plot_tests=5
dirname, filename = os.path.split(os.path.abspath(__file__))
graphpath = '\\graphs\\'
if not os.path.exists(dirname+graphpath):
    os.makedirs(dirname+graphpath)
original_wavedata=pd.read_csv('townsville-wavedata-1975-2019.csv', low_memory=False)
pd.set_option('display.float_format', lambda x:'%f'%x)
original_wavedata['Date/Time']=pd.to_datetime(original_wavedata['Date/Time'])
original_wavedata=original_wavedata.set_index(pd.DatetimeIndex(original_wavedata['Date/Time']))
columns=['Hs', 'Hmax', 'Tz', 'Tp', 'Dir_Tp TRUE', 'SST']
def normalise_dataframe(dataframe):
    x=copy.deepcopy(dataframe)
    min_max_scaler=preprocessing.MinMaxScaler()
    x_scaled=min_max_scaler.fit_transform(x)
    normalised_dataframe=pd.DataFrame(x_scaled, columns=dataframe.columns, index=dataframe.index)
    return normalised_dataframe
#Pre-processing
wavedata_adjusted=copy.deepcopy(original_wavedata)
wavedata_adjusted['Hs']=wavedata_adjusted['Hs'][wavedata_adjusted.drop(["Date/Time", "Hmax", "Tz", "Tp", "Dir_Tp TRUE", "SST"], axis=1).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
wavedata_adjusted['Hmax']=wavedata_adjusted['Hmax'][wavedata_adjusted.drop(["Date/Time", "Hs", "Tz", "Tp", "Dir_Tp TRUE", "SST"], axis=1).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
wavedata_adjusted['Tz']=wavedata_adjusted['Tz'][wavedata_adjusted.drop(["Date/Time", "Hs", "Hmax", "Tp", "Dir_Tp TRUE", "SST"], axis=1).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
wavedata_adjusted['Tp']=wavedata_adjusted['Tp'][wavedata_adjusted.drop(["Date/Time", "Hs", "Hmax", "Tz", "Dir_Tp TRUE", "SST"], axis=1).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
wavedata_adjusted=wavedata_adjusted.drop(["Date/Time", "Dir_Tp TRUE", "SST"], axis=1).dropna()
original_dir_tp_months={}
for i in range(1, 13):
    original_dir_tp_months[i]=original_wavedata[original_wavedata["Date/Time"].dt.month==i].drop(["Date/Time", "Hs", "Hmax", "Tz", "Tp", "SST"], axis=1)
original_dir_tp_summer=original_dir_tp_months[1].append(original_dir_tp_months[2].append(original_dir_tp_months[12]))
original_dir_tp_autumn=original_dir_tp_months[3].append(original_dir_tp_months[4].append(original_dir_tp_months[5]))
original_dir_tp_winter=original_dir_tp_months[6].append(original_dir_tp_months[7].append(original_dir_tp_months[8]))
original_dir_tp_spring=original_dir_tp_months[9].append(original_dir_tp_months[10].append(original_dir_tp_months[11]))
original_dir_tp_weeks={}
for i in range(1, 53):
    original_dir_tp_weeks[i]=original_wavedata[original_wavedata["Date/Time"].dt.week==i].drop(["Date/Time", "Hs", "Hmax", "Tz", "Tp", "SST"], axis=1)
original_sst_months={}
for i in range(1, 13):
    original_sst_months[i]=original_wavedata[original_wavedata["Date/Time"].dt.month==i].drop(["Date/Time", "Hs", "Hmax", "Tz", "Tp", "Dir_Tp TRUE"], axis=1)
original_sst_summer=original_sst_months[1].append(original_sst_months[2].append(original_sst_months[12]))
original_sst_autumn=original_sst_months[3].append(original_sst_months[4].append(original_sst_months[5]))
original_sst_winter=original_sst_months[6].append(original_sst_months[7].append(original_sst_months[8]))
original_sst_spring=original_sst_months[9].append(original_sst_months[10].append(original_sst_months[11]))
original_sst_weeks={}
for i in range(1, 53):
    original_sst_weeks[i]=original_wavedata[original_wavedata["Date/Time"].dt.week==i].drop(["Date/Time", "Hs", "Hmax", "Tz", "Tp", "Dir_Tp TRUE"], axis=1)
original_sst_months_day={}
for i in range(1, 13):
    original_sst_months_day[i]=original_sst_months[i].between_time('6:00', '18:00')
original_sst_summer_day=original_sst_months_day[1].append(original_sst_months_day[2].append(original_sst_months_day[12]))
original_sst_autumn_day=original_sst_months_day[3].append(original_sst_months_day[4].append(original_sst_months_day[5]))
original_sst_winter_day=original_sst_months_day[6].append(original_sst_months_day[7].append(original_sst_months_day[8]))
original_sst_spring_day=original_sst_months_day[9].append(original_sst_months_day[10].append(original_sst_months_day[11]))
original_sst_weeks_day={}
for i in range(1, 53):
    original_sst_weeks_day[i]=original_wavedata[original_wavedata["Date/Time"].dt.week==i].drop(["Date/Time", "Hs", "Hmax", "Tz", "Tp", "Dir_Tp TRUE"], axis=1).between_time('6:00', '18:00')
original_sst_months_night={}
for i in range(1, 13):
    original_sst_months_night[i]=original_sst_months[i].between_time('18:00', '6:00')
original_sst_summer_night=original_sst_months_night[1].append(original_sst_months_night[2].append(original_sst_months_night[12]))
original_sst_autumn_night=original_sst_months_night[3].append(original_sst_months_night[4].append(original_sst_months_night[5]))
original_sst_winter_night=original_sst_months_night[6].append(original_sst_months_night[7].append(original_sst_months_night[8]))
original_sst_spring_night=original_sst_months_night[9].append(original_sst_months_night[10].append(original_sst_months_night[11]))
original_sst_weeks_night={}
for i in range(1, 53):
    original_sst_weeks_night[i]=original_wavedata[original_wavedata["Date/Time"].dt.week==i].drop(["Date/Time", "Hs", "Hmax", "Tz", "Tp", "Dir_Tp TRUE"], axis=1).between_time('18:00', '6:00')
dir_tp_months_removed_outliers={}
for i in range(1, 13):
    dir_tp_months_removed_outliers[i]=original_dir_tp_months[i][original_dir_tp_months[i].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
dir_tp_months_mean={}
for i in range(1, 13):
    dir_tp_months_mean[i]=dir_tp_months_removed_outliers[i].mean()
dir_tp_months_adjusted=original_wavedata.drop(["Hs", "Hmax", "Tz", "Tp", "SST"], axis=1)
for i in range(1, 13):
    dir_tp_months_adjusted[dir_tp_months_adjusted["Date/Time"].dt.month==i]=dir_tp_months_adjusted[dir_tp_months_adjusted["Date/Time"].dt.month==i].replace(np.nan, dir_tp_months_mean[i])
for i in range(1, 13):
    dir_tp_months_adjusted[dir_tp_months_adjusted["Date/Time"].dt.month==i]=dir_tp_months_adjusted[dir_tp_months_adjusted["Date/Time"].dt.month==i][pd.DataFrame(dir_tp_months_adjusted[dir_tp_months_adjusted["Date/Time"].dt.month==i]['Dir_Tp TRUE']).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
dir_tp_months_adjusted=dir_tp_months_adjusted.drop(["Date/Time"], axis=1)
dir_tp_summer_removed_outliers=original_dir_tp_summer[original_dir_tp_summer.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
dir_tp_autumn_removed_outliers=original_dir_tp_autumn[original_dir_tp_autumn.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
dir_tp_winter_removed_outliers=original_dir_tp_winter[original_dir_tp_winter.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
dir_tp_spring_removed_outliers=original_dir_tp_spring[original_dir_tp_spring.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
dir_tp_summer_mean=dir_tp_summer_removed_outliers.mean()
dir_tp_autumn_mean=dir_tp_autumn_removed_outliers.mean()
dir_tp_winter_mean=dir_tp_winter_removed_outliers.mean()
dir_tp_spring_mean=dir_tp_spring_removed_outliers.mean()
dir_tp_summer_adjusted=original_dir_tp_summer
dir_tp_autumn_adjusted=original_dir_tp_autumn
dir_tp_winter_adjusted=original_dir_tp_winter
dir_tp_spring_adjusted=original_dir_tp_spring
dir_tp_summer_adjusted=dir_tp_summer_adjusted.replace(np.nan, dir_tp_summer_mean)
dir_tp_autumn_adjusted=dir_tp_autumn_adjusted.replace(np.nan, dir_tp_autumn_mean)
dir_tp_winter_adjusted=dir_tp_winter_adjusted.replace(np.nan, dir_tp_winter_mean)
dir_tp_spring_adjusted=dir_tp_spring_adjusted.replace(np.nan, dir_tp_spring_mean)
dir_tp_summer_adjusted=dir_tp_summer_adjusted[dir_tp_summer_adjusted.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
dir_tp_autumn_adjusted=dir_tp_autumn_adjusted[dir_tp_autumn_adjusted.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
dir_tp_winter_adjusted=dir_tp_winter_adjusted[dir_tp_winter_adjusted.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
dir_tp_spring_adjusted=dir_tp_spring_adjusted[dir_tp_spring_adjusted.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
dir_tp_season_adjusted=dir_tp_summer_adjusted.append(dir_tp_autumn_adjusted.append(dir_tp_winter_adjusted.append(dir_tp_spring_adjusted)))
dir_tp_weeks_removed_outliers={}
for i in range(1, 53):
    dir_tp_weeks_removed_outliers[i]=original_dir_tp_weeks[i][original_dir_tp_weeks[i].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
dir_tp_weeks_mean={}
for i in range(1, 53):
    dir_tp_weeks_mean[i]=dir_tp_weeks_removed_outliers[i].mean()
dir_tp_weeks_adjusted=original_wavedata.drop(["Hs", "Hmax", "Tz", "Tp", "SST"], axis=1)
for i in range(1, 53):
    dir_tp_weeks_adjusted[dir_tp_weeks_adjusted["Date/Time"].dt.week==i]=dir_tp_weeks_adjusted[dir_tp_weeks_adjusted["Date/Time"].dt.week==i].replace(np.nan, dir_tp_weeks_mean[i])
for i in range(1, 53):
    dir_tp_weeks_adjusted[dir_tp_weeks_adjusted["Date/Time"].dt.week==i]=dir_tp_weeks_adjusted[dir_tp_weeks_adjusted["Date/Time"].dt.week==i][pd.DataFrame(dir_tp_weeks_adjusted[dir_tp_weeks_adjusted["Date/Time"].dt.week==i]['Dir_Tp TRUE']).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
dir_tp_weeks_adjusted=dir_tp_weeks_adjusted.drop(["Date/Time"], axis=1)
sst_months_day_removed_outliers={}
for i in range(1, 13):
    sst_months_day_removed_outliers[i]=original_sst_months_day[i][original_sst_months_day[i].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_months_day_mean={}
for i in range(1, 13):
    sst_months_day_mean[i]=sst_months_day_removed_outliers[i].mean()
sst_months_day_adjusted=original_wavedata.drop(["Hs", "Hmax", "Tz", "Tp", "Dir_Tp TRUE"], axis=1).between_time('6:00', '18:00')
for i in range(1, 13):
    sst_months_day_adjusted[sst_months_day_adjusted["Date/Time"].dt.month==i]=sst_months_day_adjusted[sst_months_day_adjusted["Date/Time"].dt.month==i].replace(np.nan, sst_months_day_mean[i])
for i in range(1, 13):
    sst_months_day_adjusted[sst_months_day_adjusted["Date/Time"].dt.month==i]=sst_months_day_adjusted[sst_months_day_adjusted["Date/Time"].dt.month==i][pd.DataFrame(sst_months_day_adjusted[sst_months_day_adjusted["Date/Time"].dt.month==i]['SST']).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_months_day_adjusted=sst_months_day_adjusted.drop(["Date/Time"], axis=1)
sst_summer_day_removed_outliers=original_sst_summer_day[original_sst_summer_day.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_autumn_day_removed_outliers=original_sst_autumn_day[original_sst_autumn_day.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_winter_day_removed_outliers=original_sst_winter_day[original_sst_winter_day.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_spring_day_removed_outliers=original_sst_spring_day[original_sst_spring_day.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_summer_day_mean=sst_summer_day_removed_outliers.mean()
sst_autumn_day_mean=sst_autumn_day_removed_outliers.mean()
sst_winter_day_mean=sst_winter_day_removed_outliers.mean()
sst_spring_day_mean=sst_spring_day_removed_outliers.mean()
sst_summer_day_adjusted=original_sst_summer_day
sst_autumn_day_adjusted=original_sst_autumn_day
sst_winter_day_adjusted=original_sst_winter_day
sst_spring_day_adjusted=original_sst_spring_day
sst_summer_day_adjusted=sst_summer_day_adjusted.replace(np.nan, sst_summer_day_mean)
sst_autumn_day_adjusted=sst_autumn_day_adjusted.replace(np.nan, sst_autumn_day_mean)
sst_winter_day_adjusted=sst_winter_day_adjusted.replace(np.nan, sst_winter_day_mean)
sst_spring_day_adjusted=sst_spring_day_adjusted.replace(np.nan, sst_spring_day_mean)
sst_summer_day_adjusted=sst_summer_day_adjusted[sst_summer_day_adjusted.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_autumn_day_adjusted=sst_autumn_day_adjusted[sst_autumn_day_adjusted.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_winter_day_adjusted=sst_winter_day_adjusted[sst_winter_day_adjusted.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_spring_day_adjusted=sst_spring_day_adjusted[sst_spring_day_adjusted.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_season_day_adjusted=sst_summer_day_adjusted.append(sst_autumn_day_adjusted.append(sst_winter_day_adjusted.append(sst_spring_day_adjusted)))
sst_weeks_day_removed_outliers={}
for i in range(1, 53):
    sst_weeks_day_removed_outliers[i]=original_sst_weeks_day[i][original_sst_weeks_day[i].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_weeks_day_mean={}
for i in range(1, 53):
    sst_weeks_day_mean[i]=sst_weeks_day_removed_outliers[i].mean()
sst_weeks_day_adjusted=original_wavedata.drop(["Hs", "Hmax", "Tz", "Tp", "Dir_Tp TRUE"], axis=1).between_time('6:00', '18:00')
for i in range(1, 53):
    sst_weeks_day_adjusted[sst_weeks_day_adjusted["Date/Time"].dt.week==i]=sst_weeks_day_adjusted[sst_weeks_day_adjusted["Date/Time"].dt.week==i].replace(np.nan, sst_weeks_day_mean[i])
for i in range(1, 53):
    sst_weeks_day_adjusted[sst_weeks_day_adjusted["Date/Time"].dt.month==i]=sst_weeks_day_adjusted[sst_weeks_day_adjusted["Date/Time"].dt.month==i][pd.DataFrame(sst_weeks_day_adjusted[sst_weeks_day_adjusted["Date/Time"].dt.month==i]['SST']).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_weeks_day_adjusted=sst_weeks_day_adjusted.drop(["Date/Time"], axis=1)
sst_months_night_removed_outliers={}
for i in range(1, 13):
    sst_months_night_removed_outliers[i]=original_sst_months_night[i][original_sst_months_night[i].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_months_night_mean={}
for i in range(1, 13):
    sst_months_night_mean[i]=sst_months_night_removed_outliers[i].mean()
sst_months_night_adjusted=original_wavedata.drop(["Hs", "Hmax", "Tz", "Tp", "Dir_Tp TRUE"], axis=1).between_time('18:00', '6:00')
for i in range(1, 13):
    sst_months_night_adjusted[sst_months_night_adjusted["Date/Time"].dt.month==i]=sst_months_night_adjusted[sst_months_night_adjusted["Date/Time"].dt.month==i].replace(np.nan, sst_months_night_mean[i])
for i in range(1, 13):
    sst_months_night_adjusted[sst_months_night_adjusted["Date/Time"].dt.month==i]=sst_months_night_adjusted[sst_months_night_adjusted["Date/Time"].dt.month==i][pd.DataFrame(sst_months_night_adjusted[sst_months_night_adjusted["Date/Time"].dt.month==i]['SST']).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_months_night_adjusted=sst_months_night_adjusted.drop(["Date/Time"], axis=1)
sst_summer_night_removed_outliers=original_sst_summer_night[original_sst_summer_night.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_autumn_night_removed_outliers=original_sst_autumn_night[original_sst_autumn_night.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_winter_night_removed_outliers=original_sst_winter_night[original_sst_winter_night.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_spring_night_removed_outliers=original_sst_spring_night[original_sst_spring_night.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_summer_night_mean=sst_summer_night_removed_outliers.mean()
sst_autumn_night_mean=sst_autumn_night_removed_outliers.mean()
sst_winter_night_mean=sst_winter_night_removed_outliers.mean()
sst_spring_night_mean=sst_spring_night_removed_outliers.mean()
sst_summer_night_adjusted=original_sst_summer_night
sst_autumn_night_adjusted=original_sst_autumn_night
sst_winter_night_adjusted=original_sst_winter_night
sst_spring_night_adjusted=original_sst_spring_night
sst_summer_night_adjusted=sst_summer_night_adjusted.replace(np.nan, sst_summer_night_mean)
sst_autumn_night_adjusted=sst_autumn_night_adjusted.replace(np.nan, sst_autumn_night_mean)
sst_winter_night_adjusted=sst_winter_night_adjusted.replace(np.nan, sst_winter_night_mean)
sst_spring_night_adjusted=sst_spring_night_adjusted.replace(np.nan, sst_spring_night_mean)
sst_summer_night_adjusted=sst_summer_night_adjusted[sst_summer_night_adjusted.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_autumn_night_adjusted=sst_autumn_night_adjusted[sst_autumn_night_adjusted.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_winter_night_adjusted=sst_winter_night_adjusted[sst_winter_night_adjusted.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_spring_night_adjusted=sst_spring_night_adjusted[sst_spring_night_adjusted.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_season_night_adjusted=sst_summer_night_adjusted.append(sst_autumn_night_adjusted.append(sst_winter_night_adjusted.append(sst_spring_night_adjusted)))
sst_weeks_night_removed_outliers={}
for i in range(1, 53):
    sst_weeks_night_removed_outliers[i]=original_sst_weeks_night[i][original_sst_weeks_night[i].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_weeks_night_mean={}
for i in range(1, 53):
    sst_weeks_night_mean[i]=sst_weeks_night_removed_outliers[i].mean()
sst_weeks_night_adjusted=original_wavedata.drop(["Hs", "Hmax", "Tz", "Tp", "Dir_Tp TRUE"], axis=1).between_time('18:00', '6:00')
for i in range(1, 53):
    sst_weeks_night_adjusted[sst_weeks_night_adjusted["Date/Time"].dt.week==i]=sst_weeks_night_adjusted[sst_weeks_night_adjusted["Date/Time"].dt.week==i].replace(np.nan, sst_weeks_night_mean[i])
for i in range(1, 53):
    sst_weeks_night_adjusted[sst_weeks_night_adjusted["Date/Time"].dt.month==i]=sst_weeks_night_adjusted[sst_weeks_night_adjusted["Date/Time"].dt.month==i][pd.DataFrame(sst_weeks_night_adjusted[sst_weeks_night_adjusted["Date/Time"].dt.month==i]['SST']).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
sst_weeks_night_adjusted=sst_weeks_night_adjusted.drop(["Date/Time"], axis=1)
sst_months_adjusted=sst_months_day_adjusted.append(sst_months_night_adjusted).dropna()
sst_season_adjusted=sst_season_day_adjusted.append(sst_season_night_adjusted).dropna()
sst_weeks_adjusted=sst_weeks_day_adjusted.append(sst_weeks_night_adjusted).dropna()
wavedata_dir_tp_months_sst_months=pd.merge(wavedata_adjusted, dir_tp_months_adjusted.join(sst_months_adjusted), 'right', right_index=True, left_on='Date/Time').dropna()
wavedata_dir_tp_months_sst_season=pd.merge(wavedata_adjusted, dir_tp_months_adjusted.join(sst_season_adjusted), 'right', right_index=True, left_on='Date/Time').dropna()
wavedata_dir_tp_months_sst_weeks=pd.merge(wavedata_adjusted, dir_tp_months_adjusted.join(sst_weeks_adjusted), 'right', right_index=True, left_on='Date/Time').dropna()
wavedata_dir_tp_season_sst_months=pd.merge(wavedata_adjusted, dir_tp_season_adjusted.join(sst_months_adjusted), 'right', right_index=True, left_on='Date/Time').dropna()
wavedata_dir_tp_season_sst_season=pd.merge(wavedata_adjusted, dir_tp_season_adjusted.join(sst_season_adjusted), 'right', right_index=True, left_on='Date/Time').dropna()
wavedata_dir_tp_season_sst_weeks=pd.merge(wavedata_adjusted, dir_tp_season_adjusted.join(sst_weeks_adjusted), 'right', right_index=True, left_on='Date/Time').dropna()
wavedata_dir_tp_weeks_sst_months=pd.merge(wavedata_adjusted, dir_tp_weeks_adjusted.join(sst_months_adjusted), 'right', right_index=True, left_on='Date/Time').dropna()
wavedata_dir_tp_weeks_sst_season=pd.merge(wavedata_adjusted, dir_tp_weeks_adjusted.join(sst_season_adjusted), 'right', right_index=True, left_on='Date/Time').dropna()
wavedata_dir_tp_weeks_sst_weeks=pd.merge(wavedata_adjusted, dir_tp_weeks_adjusted.join(sst_weeks_adjusted), 'right', right_index=True, left_on='Date/Time').dropna()
dataset_combinations={}
dataset_combinations['months_months']=normalised_wavedata_dir_tp_months_sst_months=normalise_dataframe(wavedata_dir_tp_months_sst_months).sort_index().dropna()
dataset_combinations['months_season']=normalised_wavedata_dir_tp_months_sst_season=normalise_dataframe(wavedata_dir_tp_months_sst_season).sort_index().dropna()
dataset_combinations['months_weeks']=normalised_wavedata_dir_tp_months_sst_weeks=normalise_dataframe(wavedata_dir_tp_months_sst_weeks).sort_index().dropna()
dataset_combinations['season_months']=normalised_wavedata_dir_tp_season_sst_months=normalise_dataframe(wavedata_dir_tp_season_sst_months).sort_index().dropna()
dataset_combinations['season_season']=normalised_wavedata_dir_tp_season_sst_season=normalise_dataframe(wavedata_dir_tp_season_sst_season).sort_index().dropna()
dataset_combinations['season_weeks']=normalised_wavedata_dir_tp_season_sst_weeks=normalise_dataframe(wavedata_dir_tp_season_sst_weeks).sort_index().dropna()
dataset_combinations['weeks_months']=normalised_wavedata_dir_tp_weeks_sst_months=normalise_dataframe(wavedata_dir_tp_weeks_sst_months).sort_index().dropna()
dataset_combinations['weeks_season']=normalised_wavedata_dir_tp_weeks_sst_season=normalise_dataframe(wavedata_dir_tp_weeks_sst_season).sort_index().dropna()
dataset_combinations['weeks_weeks']=normalised_wavedata_dir_tp_weeks_sst_weeks=normalise_dataframe(wavedata_dir_tp_weeks_sst_weeks).sort_index().dropna()
#Datamining
def kmeans_clustering_initial(k=1):
    # Initial points
    centroids={}
    centroids_step = {
    i+1: [np.random.random(), np.random.random()]
    for i in range(k)
    }
    centroids[0]=centroids_step
    return centroids
def euclidean_distance(x, cx, y, cy):
    distance=np.sqrt((x-cx)**2 + (y-cy)**2)
    return distance
def deviation(x, cx, y, cy):
    deviation=(x-cx)+(y-cy)
    return deviation
def kmeans_clustering_assignment(dataframe, x, y, centroids, k, i):
    assignment=copy.deepcopy(dataframe)
    for ik in range(1, k+1):
        assignment['distance_from_{}'.format(ik)]=euclidean_distance(dataframe[x], centroids[i][ik][0], dataframe[y], centroids[i][ik][0])
        assignment['deviation_from_{}'.format(ik)]=deviation(dataframe[x], centroids[i][ik][0], dataframe[y], centroids[i][ik][0])
    centroid_distance_cols=['distance_from_{}'.format(ik) for ik in centroids[i].keys()]
    assignment['closest']=assignment.loc[:, centroid_distance_cols].idxmin(axis=1)
    assignment['closest']=assignment['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    return assignment
def kmeans_clustering_update_centroids(dataset, x, y, k, i, centroids):
    centroids[i]=copy.deepcopy(centroids[i-1])
    for ik in range(1, k+1):
        centroids[i][ik]=[np.mean(dataset[dataset['closest']==ik][x]), np.mean(dataset[dataset['closest']==ik][y])]
    return centroids
def within_cluster_sum_of_square_errors(dataset, k, i):
    wss=0.0
    for ik in range(1, k+1):
        temp=0.0
        cluster=dataset['closest']==ik
        cluster_data=dataset[cluster]['deviation_from_{}'.format(ik)].replace(np.nan, 1.0)
        count=cluster_data.count()
        cluster_data_2=np.power(cluster_data, 2)
        cluster_sum=cluster_data_2.sum()
        if cluster_sum==0.0:
            temp=0.0
        else:
            temp=cluster_sum/count
        wss+=temp
    return wss
def silhouette_value(dataset, x, y, k, i):
    no_null_dataset=dataset.replace(np.nan, 1.0)
    cluster_nums=no_null_dataset['closest'].unique().tolist()
    a=[]
    b=[]
    for row in no_null_dataset.itertuples():
        distances={}
        for jk in cluster_nums:
            if jk==1:
                row_num=6+jk
            else:
                row_num=6+jk+(int(jk-1))
            distances['distance_from_{}'.format(jk)]=row[row_num]
        sorted_distances=sorted(distances.items(), key=lambda kv: kv[1])
        sorted_distances=collections.OrderedDict(sorted_distances)
        ai_key=list(sorted_distances)[0]
        try:
            bi_key=list(sorted_distances)[1]
        except:
            return 0
        ai=sorted_distances[ai_key]
        bi=sorted_distances[bi_key]
        if ai==np.nan:
            ai=1.0
        if bi==np.nan:
            bi=1.0
        a.append(ai)
        b.append(bi)
    a=np.array(a)
    b=np.array(b)
    SSI=(b-a)/np.maximum(a, b)
    SSI=SSI.sum()/len(SSI)
    #print('{} - {}'.format(k, SSI))
    return SSI
def no_centroid_change(centroids, k, i):
    result=False
    for ik in range(1, k+1):
        x=centroids[i][ik][0]==centroids[i-1][ik][0]
        y=centroids[i][ik][1]==centroids[i-1][ik][1]
        if x==False or y==False:
            result=True
    return result
def kmeans_clustering(dataframe, x, y, k=1):
    kmeans=copy.deepcopy(dataframe)
    centroids=kmeans_clustering_initial(k)
    kmeans=kmeans_clustering_assignment(kmeans, x, y, centroids, k, 0)
    i=0
    while True:
        i+=1
        centroids = kmeans_clustering_update_centroids(kmeans, x, y, k, i, centroids)
        kmeans = kmeans_clustering_assignment(kmeans, x, y, centroids, k, i)
        if no_centroid_change(centroids, k, i):
            break;
    return kmeans, silhouette_value(kmeans, x, y, k, i)
def do_kmeans(dataframe, x, y):
    plots={}
    count=0
    total_plot_time=0
    for i in range(2, maximum_k+1):
        j_plots={}
        for j in range(0, number_of_plot_tests):
            start = time.time()
            j_plot, j_ssi=kmeans_clustering(dataframe, x, y, i)
            j_plots[j_ssi]=j_plot
            end = time.time()
            elapsed = end - start
            total_plot_time+=elapsed
            count+=1
            #print("Plot {} - time elapsed - {} - total time elapsed - {} - wss - {}".format(count, elapsed, total_plot_time, j_wss))
        ssi = sorted(j_plots, reverse=True)[0]
        #print(wss)
        plot = j_plots[ssi]
        plots[ssi] = plot
    #print("Total time elapsed - {}".format(total_plot_time))
    return plots[sorted(plots)[0]], total_plot_time
columns=original_wavedata.columns.drop('Date/Time')
column_combinations=[]
column_combinations=list(itertools.combinations(columns, 2))
plots_to_calculate=[]
column_combos_count={}
for dataset_key, dataset in dataset_combinations.items():
    for combo in column_combinations:
        x=dataset[combo[0]]
        y=dataset[combo[1]]
        # result[0] is a value between -1 and 1
        # The null hypothesis is that the two columns are not correlated
        # The result is a number between 0 and one that represents the probability
        # that the data would have arisen if the null hypothesis is true
        result=scipy.stats.kendalltau(x, y)[0]
        x_r=result>0.5 or result<-0.5
        if(x_r):
            plots_to_calculate.append((dataset_key, dataset, combo[0], combo[1]))
            if '{}-{}'.format(combo[0], combo[1]) in column_combos_count:
                column_combos_count['{}-{}'.format(combo[0], combo[1])]+=1
            else:
                column_combos_count['{}-{}'.format(combo[0], combo[1])]=1
            #print('{}-{}-{}'.format(dataset_key, combo[0], combo[1]))
print(len(plots_to_calculate))
for key, value in column_combos_count.items():
    print('{}-{}'.format(key, value))
old_plots_to_calculate=plots_to_calculate
hashplots={}
for entry in plots_to_calculate:
    dataset_key=entry[0]
    dataset=entry[1]
    x=entry[2]
    y=entry[3]
    xhash=hash(dataset[x].values.tostring())
    yhash=hash(dataset[y].values.tostring())
    thash=xhash+yhash
    hashplots[thash]=entry
plots_to_calculate=[]
for key, entry in hashplots.items():
    plots_to_calculate.append(entry)
print('Plots to calculate - {}'.format(len(plots_to_calculate)))
print(hashplots.keys())
for plot in plots_to_calculate:
    print('{}-{}-{}'.format(plot[0], plot[2], plot[3]))
plots={}
total_time=0
num=len(plots_to_calculate)
num_left=num
time_left_estimate=total_time
for entry in plots_to_calculate:
    dataset_key=entry[0]
    dataset=entry[1]
    x=entry[2]
    y=entry[3]
    plots['{}_{}_{}'.format(dataset_key, x, y)], plot_time=do_kmeans(dataset, x, y)
    total_time+=plot_time
    num_left-=1
    average_completion_time=total_time/(num-num_left)
    time_left_estimate=(num_left*(average_completion_time))
    print('Completed {}/{} plots. Average plot completion time is {}. Estimated time remaining is {}'.format((num-num_left), num, average_completion_time, time_left_estimate))
#Create graphs
graphs={}
plt.close('all')
for entry in plots_to_calculate:
    dataset_key=entry[0]
    dataset=entry[1]
    x=entry[2]
    y=entry[3]
    graphs['{}_{}_{}'.format(dataset_key, x, y)]=sns.lmplot(x, y, data=plots['{}_{}_{}'.format(dataset_key, x, y)], fit_reg=False, hue='closest')
    axes = graphs['{}_{}_{}'.format(dataset_key, x, y)].axes.flatten()
    axes[0].set_title('{} {} vs {}'.format(dataset_key, x, y))
    graphs['{}_{}_{}'.format(dataset_key, x, y)].savefig(dirname+graphpath+'{}_{}_{}.png'.format(x, y, dataset_key))
    #graphs['{}_{}_{}'.format(dataset_key, x, y)].close()
plt.close('all')
