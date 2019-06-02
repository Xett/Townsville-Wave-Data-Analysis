import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import copy
import time
import itertools
import scipy
from sklearn.metrics import silhouette_samples, silhouette_score
import collections
np.random.seed(12345)
maximum_k=7
number_of_plot_tests=5
columns=['Hs', 'Hmax', 'Tz', 'Tp', 'Dir_Tp TRUE', 'SST']
column_combinations=list(itertools.combinations(columns, 2))

def get_columns():
    return copy.deepcopy(columns)

def load_dataset():
    original_wavedata=pd.read_csv('data files/townsville-wavedata-1975-2019.csv', low_memory=False)
    pd.set_option('display.float_format', lambda x:'%f'%x)
    original_wavedata['Date/Time']=pd.to_datetime(original_wavedata['Date/Time'])
    original_wavedata=original_wavedata.set_index(pd.DatetimeIndex(original_wavedata['Date/Time']))
    print('Loaded Dataset')
    return original_wavedata

def normalise_dataframe(dataframe):
    x=copy.deepcopy(dataframe)
    min_max_scaler=preprocessing.MinMaxScaler()
    x_scaled=min_max_scaler.fit_transform(x)
    normalised_dataframe=pd.DataFrame(x_scaled, columns=dataframe.columns, index=dataframe.index)
    return normalised_dataframe

def remove_outliers(original_wavedata):
    wavedata_adjusted=copy.deepcopy(original_wavedata)
    wavedata_adjusted['Hs']=wavedata_adjusted['Hs'][wavedata_adjusted.drop(["Date/Time", "Hmax", "Tz", "Tp", "Dir_Tp TRUE", "SST"], axis=1).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
    wavedata_adjusted['Hmax']=wavedata_adjusted['Hmax'][wavedata_adjusted.drop(["Date/Time", "Hs", "Tz", "Tp", "Dir_Tp TRUE", "SST"], axis=1).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
    wavedata_adjusted['Tz']=wavedata_adjusted['Tz'][wavedata_adjusted.drop(["Date/Time", "Hs", "Hmax", "Tp", "Dir_Tp TRUE", "SST"], axis=1).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
    wavedata_adjusted['Tp']=wavedata_adjusted['Tp'][wavedata_adjusted.drop(["Date/Time", "Hs", "Hmax", "Tz", "Dir_Tp TRUE", "SST"], axis=1).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
    wavedata_adjusted=wavedata_adjusted.drop(["Date/Time", "Dir_Tp TRUE", "SST"], axis=1).dropna()
    print('Removed Outliers')
    return wavedata_adjusted

# Variable Declarations
def create_months(dataset, output_dict, column, mode):
    _columns=get_columns()
    _columns.append("Date/Time")
    _columns.remove(column)
    time1='0:00'
    time2='23:59'
    dataset_selection=None
    if mode==1:
        time1='6:00'
        time2='18:00'
    elif mode==2:
        time1='18:00'
        time2='6:00'
    for i in range(1, 13):
        if mode==0:
            dataset_selection=dataset[dataset["Date/Time"].dt.month==i].drop(_columns, axis=1)
        else:
            dataset_selection=dataset[i]
        output_dict[i]=dataset_selection.between_time(time1, time2)
    if mode==0:
        print('Created Months Dictionary for Column {}'.format(column))
    elif mode==1:
        print('Created Months Dictionary for Column {}_day'.format(column))
    elif mode==2:
        print('Created Months Dictionary for Column {}_night'.format(column))
    return output_dict

def create_seasons(month_dictionary, output_dict, column, mode):
    output_dict[1]=month_dictionary[1].append(month_dictionary[2].append(month_dictionary[12]))
    output_dict[2]=month_dictionary[3].append(month_dictionary[4].append(month_dictionary[5]))
    output_dict[3]=month_dictionary[6].append(month_dictionary[7].append(month_dictionary[8]))
    output_dict[4]=month_dictionary[9].append(month_dictionary[10].append(month_dictionary[11]))
    if mode==0:
        print('Created Seasons Dictionary for Column {}'.format(column))
    elif mode==1:
        print('Created Seasons Dictionary for Column {}_day'.format(column))
    elif mode==2:
        print('Created Seasons Dictionary for Column {}_night'.format(column))
    return output_dict

def create_weeks(dataset, output_dict, column, mode):
    _columns=get_columns()
    _columns.append("Date/Time")
    _columns.remove(column)
    time1='0:00'
    time2='23:59'
    if mode==1:
        time1='6:00'
        time2='18:00'
    elif mode==2:
        time1='18:00'
        time2='6:00'
    for i in range(1, 53):
        if mode==0:
            dataset_selection=dataset[dataset["Date/Time"].dt.week==i].drop(_columns, axis=1)
        else:
            dataset_selection=dataset[i]
        output_dict[i]=dataset_selection.between_time(time1, time2)
    if mode==0:
        print('Created Weeks Dictionary for Column {}'.format(column))
    elif mode==1:
        print('Created Weeks Dictionary for Column {}_day'.format(column))
    elif mode==2:
        print('Created Weeks Dictionary for Column {}_night'.format(column))
    return output_dict

# remove outliers and fill null values
def remove_outliers_fill_nulls_months(months_dataset, original_wavedata, column, mode):
    _columns=get_columns()
    _columns.remove(column)
    removed_outliers={}
    months_mean={}
    months_adjusted=original_wavedata.drop(_columns, axis=1)
    for i in range(1, 13):
        removed_outliers[i]=months_dataset[i][months_dataset[i].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
        months_mean[i]=removed_outliers[i].mean()
        months_adjusted[months_adjusted["Date/Time"].dt.month==i]=months_adjusted[months_adjusted["Date/Time"].dt.month==i].replace(np.nan, months_mean[i])
        months_adjusted[months_adjusted["Date/Time"].dt.month==i]=months_adjusted[months_adjusted["Date/Time"].dt.month==i][pd.DataFrame(months_adjusted[months_adjusted["Date/Time"].dt.month==i][column]).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
    months_adjusted=months_adjusted.drop(["Date/Time"], axis=1)
    if mode==0:
        print('Removed Outliers and Filled Months for Column {}'.format(column))
    elif mode==1:
        print('Removed Outliers and Filled Months for Column {}_day'.format(column))
    elif mode==2:
        print('Removed Outliers and Filled Months for Column {}_night'.format(column))
    return months_adjusted

def remove_outliers_fill_nulls_seasons(seasons_dict, months_dict, column, mode):
    _columns=get_columns()
    _columns.remove(column)
    removed_outliers={}
    seasons_mean={}
    seasons_adjusted={}
    for i in range(1, 5):
        removed_outliers[i]=seasons_dict[i][seasons_dict[i].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
        seasons_mean[i]=removed_outliers[i].mean()
        seasons_adjusted[i]=months_dict[i]
        seasons_adjusted[i]=seasons_adjusted[i].replace(np.nan, seasons_mean[i])
        seasons_adjusted[i]=seasons_adjusted[i][seasons_adjusted[i].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
    seasons_adjusted=seasons_adjusted[1].append(seasons_adjusted[2].append(seasons_adjusted[3].append(seasons_adjusted[4])))
    if mode==0:
        print('Removed Outliers and Filled Seasons for Column {}'.format(column))
    elif mode==1:
        print('Removed Outliers and Filled Seasons for Column {}_day'.format(column))
    elif mode==2:
        print('Removed Outliers and Filled Seasons for Column {}_night'.format(column))
    return seasons_adjusted

def remove_outliers_fill_nulls_weeks(weeks_dataset, original_wavedata, column, mode):
    _columns=get_columns()
    _columns.remove(column)
    removed_outliers={}
    weeks_mean={}
    weeks_adjusted=original_wavedata.drop(_columns, axis=1)
    for i in range(1, 53):
        removed_outliers[i]=weeks_dataset[i][weeks_dataset[i].apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
        weeks_mean[i]=removed_outliers[i].mean()
        weeks_adjusted[weeks_adjusted["Date/Time"].dt.month==i]=weeks_adjusted[weeks_adjusted["Date/Time"].dt.week==i].replace(np.nan, weeks_mean[i])
        weeks_adjusted[weeks_adjusted["Date/Time"].dt.month==i]=weeks_adjusted[weeks_adjusted["Date/Time"].dt.week==i][pd.DataFrame(weeks_adjusted[weeks_adjusted["Date/Time"].dt.week==i][column]).apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
    weeks_adjusted=weeks_adjusted.drop(["Date/Time"], axis=1)
    if mode==0:
        print('Removed Outliers and Filled Weeks for Column {}'.format(column))
    elif mode==1:
        print('Removed Outliers and Filled Weeks for Column {}_day'.format(column))
    elif mode==2:
        print('Removed Outliers and Filled Weeks for Column {}_night'.format(column))
    return weeks_adjusted

def create_completed_dataframes(wavedata_adjusted, datasets1, datasets2):
    completed_dataframes = {}
    for dataset1_key, dataset1 in datasets1.items():
        for dataset2_key, dataset2 in datasets2.items():
            completed_dataframes['{}_{}'.format(dataset1_key, dataset2_key)] = pd.merge(wavedata_adjusted, dataset1.join(dataset2), 'right', right_index=True, left_on='Date/Time').dropna()
    return completed_dataframes

def create_normalised_dataframes(completed_dataframes):
    normalised_dataframes={}
    for key, dataframe in completed_dataframes.items():
        normalised_dataframes[key]=normalise_dataframe(dataframe).sort_index().dropna()
    return normalised_dataframes

# Algorithm Functions
    
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

def correlate_coefficients(datasets):
    plots_to_calculate=[]
    column_combos_count={}
    for dataset_key, dataset in datasets.items():
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
    return plots_to_calculate

def calculate_plots(plots_to_calculate):
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
    return plots

def create_graphs(plots):
    graphs={}
    plt.close('all')
    for entry in plots:
        dataset_key=entry[0]
        dataset=entry[1]
        x=entry[2]
        y=entry[3]
        graphs['{}_{}_{}'.format(dataset_key, x, y)]=sns.lmplot(x, y, data=plots['{}_{}_{}'.format(dataset_key, x, y)], fit_reg=False, hue='closest')
        axes = graphs['{}_{}_{}'.format(dataset_key, x, y)].axes.flatten()
        axes[0].set_title('{} {} vs {}'.format(dataset_key, x, y))
        graphs['{}_{}_{}'.format(dataset_key, x, y)].savefig('graphs/{}_{}_{}.png'.format(x, y, dataset_key))
        #graphs['{}_{}_{}'.format(dataset_key, x, y)].close()
    plt.close('all')

# run it all
print('Starting...')
original_wavedata = load_dataset()
wavedata_adjusted = remove_outliers(original_wavedata)

print('\nStarting Pre-Processing Stage')
# Datasets before they have gone through the pre-processing stage
print('\nDir_Tp TRUE')
original_dir_tp_months = create_months(original_wavedata, {}, 'Dir_Tp TRUE', 0)
original_dir_tp_seasons = create_seasons(original_dir_tp_months, {}, 'Dir_Tp TRUE', 0)
original_dir_tp_weeks = create_weeks(original_wavedata, {}, 'Dir_Tp TRUE', 0)

print('\nSST')
original_sst_months = create_months(original_wavedata, {}, 'SST', 0)
original_sst_seasons = create_seasons(original_sst_months, {}, 'SST', 0)
original_sst_weeks = create_weeks(original_wavedata, {}, 'SST', 0)

original_sst_months_day = create_months(original_sst_months, {}, 'SST', 1)
original_sst_months_night = create_months(original_sst_months, {}, 'SST', 2)

original_sst_seasons_day = create_seasons(original_sst_months_day, {}, 'SST', 1)
original_sst_seasons_night = create_seasons(original_sst_months_night, {}, 'SST', 2)

original_sst_weeks_day = create_weeks(original_sst_weeks, {}, 'SST', 1)
original_sst_weeks_night = create_weeks(original_sst_weeks, {}, 'SST', 2)

# Remove outliers
print('\nRemove Outliers and Fill Null Values')

print('\nDir_Tp TRUE')
dir_tp_months_adjusted = remove_outliers_fill_nulls_months(original_dir_tp_months, original_wavedata, 'Dir_Tp TRUE', 0)
dir_tp_seasons_adjusted = remove_outliers_fill_nulls_seasons(original_dir_tp_seasons, original_dir_tp_months, 'Dir_Tp TRUE', 0)
dir_tp_weeks_adjusted = remove_outliers_fill_nulls_weeks(original_dir_tp_weeks, original_wavedata, 'Dir_Tp TRUE', 0)

print('\nSST')
sst_months_adjusted_day = remove_outliers_fill_nulls_months(original_sst_months_day, original_wavedata, 'SST', 1)
sst_months_adjusted_night = remove_outliers_fill_nulls_months(original_sst_months_night, original_wavedata, 'SST', 2)

sst_seasons_adjusted_day = remove_outliers_fill_nulls_seasons(original_sst_seasons_day, original_sst_months_day, 'SST', 1)
sst_seasons_adjusted_night = remove_outliers_fill_nulls_seasons(original_sst_seasons_night, original_sst_months_night, 'SST', 2)

sst_weeks_adjusted_day = remove_outliers_fill_nulls_weeks(original_sst_weeks_day, original_wavedata, 'SST', 1)
sst_weeks_adjusted_night = remove_outliers_fill_nulls_weeks(original_sst_weeks_night, original_wavedata, 'SST', 2)

print('Compile Adjusted SST Frames')
sst_months_adjusted = pd.concat([sst_months_adjusted_day, sst_months_adjusted_night], axis=1).dropna()
sst_seasons_adjusted = pd.concat([sst_seasons_adjusted_day, sst_seasons_adjusted_night], axis=1).dropna()
sst_weeks_adjusted = pd.concat([sst_weeks_adjusted_day, sst_weeks_adjusted_night], axis=1).dropna()

print('\nCreate Completed Dataframes')
dir_tp_datasets = { 'months':dir_tp_months_adjusted,
                    'seasons':dir_tp_seasons_adjusted,
                    'weeks':dir_tp_weeks_adjusted}
sst_datasets = { 'months':sst_months_adjusted,
                 'seasons':sst_seasons_adjusted,
                 'weeks':sst_weeks_adjusted}
completed_dataframes = create_completed_dataframes(wavedata_adjusted, dir_tp_datasets, sst_datasets)

print('\nNormalise Completed Dataframes')
normalised_dataframes = create_normalised_dataframes(completed_dataframes)

print('\nCorrelate Coefficients to Rule Out Unneeded Combinations')
plots_to_calculate = correlate_coefficients(normalised_dataframes)

print('\nCalculate Plots')
plots = calculate_plots(plots_to_calculate)

print('\nCreate Graphs')
create_graphs(plots)

print('Finished!')
