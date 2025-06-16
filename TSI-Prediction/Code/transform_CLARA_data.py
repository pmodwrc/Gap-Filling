import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random
from shapely.geometry import Polygon, MultiPolygon
from utils import *
import time as t
from scipy.io import readsav

PATH_OLR = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/CLARA OLR/OLR/'
PATH_POS = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/CLARA OLR/Position/'
PATH_HOUSEKEEPING = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/CLARA_combined_data.pkl'
PATH_TARGET = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/CLARA_OLR_combined.pkl'
PATH_OUT = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/CLARA_combined_all.pkl'
PATH_HIST = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Images/CLARA_histogram.png'
angle = 120
histogram = True

t0 = t.time()

if not os.path.exists(PATH_TARGET):
    df_olr = pd.DataFrame()
    df_pos = pd.DataFrame()
    olr_files = sorted([f for f in os.listdir(PATH_OLR) if os.path.isfile(os.path.join(PATH_OLR, f)) and not f == '.DS_Store'])
    pos_files = sorted([f for f in os.listdir(PATH_POS) if os.path.isfile(os.path.join(PATH_POS, f)) and not f == '.DS_Store'])
    # combine the data
    for file in olr_files:
        data = readsav(PATH_OLR + file)['olr_all']
        df_tmp = pd.DataFrame({'jday': np.concatenate(data['jday']), 'CLARA_radiance': np.concatenate(data['olr'])})
        df_tmp['CLARA_OLR_time'] = pd.to_datetime(df_tmp['jday'], origin='julian', unit='D')
        df_tmp = df_tmp.drop('jday', axis = 1)
        df_olr = pd.concat([df_olr,df_tmp], axis = 0)
    for file in pos_files:
        tmp = pd.read_pickle(PATH_POS + file)
        df_pos = pd.concat([df_pos,tmp], axis = 0)
    df_pos = df_pos.sort_values(by='CLARA_time_utc')
    df_olr = df_olr.sort_values(by='CLARA_OLR_time')
    df = pd.merge_asof(df_olr, df_pos, left_on='CLARA_OLR_time', right_on='CLARA_time_utc', direction = 'nearest')
    df = df.drop('CLARA_time_utc', axis=1)
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        # convert coordinates array to 3 columns
        if 'eci' in col and len(df[col].iloc[0]) == 3:
            df[col + '_x'] = df[col].apply(lambda x: float(x[0]) if isinstance(x, (list, np.ndarray)) else np.nan)
            df[col + '_y'] = df[col].apply(lambda x: float(x[1]) if isinstance(x, (list, np.ndarray)) else np.nan)
            df[col + '_z'] = df[col].apply(lambda x: float(x[2]) if isinstance(x, (list, np.ndarray)) else np.nan)
            df = df.drop(col, axis = 1)
        # extract features from polygon, multigon
        elif isinstance(df[col].iloc[0], (Polygon, MultiPolygon)):
            df[col + '_area'] = df[col].apply(lambda x: x.area if isinstance(x, Polygon) else np.mean([p.area for p in x.geoms]) if isinstance(x, MultiPolygon) else np.nan)
            df[col + '_length'] = df[col].apply(lambda x: x.length if isinstance(x, Polygon) else np.mean([p.length for p in x.geoms]) if isinstance(x, MultiPolygon) else np.nan)
            df[col + '_centroid_x'] = df[col].apply(lambda x: x.centroid.x if isinstance(x, Polygon) else np.mean([p.centroid.x for p in x.geoms]) if isinstance(x, MultiPolygon) else np.nan)
            df[col + '_centroid_y'] = df[col].apply(lambda x: x.centroid.y if isinstance(x, Polygon) else np.mean([p.centroid.y for p in x.geoms]) if isinstance(x, MultiPolygon) else np.nan)
            df[col + '_vertices'] = df[col].apply(lambda x: len(x.exterior.coords) if isinstance(x, Polygon) else int(np.mean([len(p.exterior.coords) for p in x.geoms])) if isinstance(x, MultiPolygon) else np.nan)
            df = df.drop(col, axis=1)
        else: 
            raise ValueError("Further preprocessing may be necessary")
    df.to_pickle(PATH_TARGET)
else:
    df = pd.read_pickle(PATH_TARGET)
features = df.columns
obj_cols = df.select_dtypes(include=['object']).columns
print('Number of features: ', len(features))
print('out of which ', len(obj_cols), ' are of type object.')
house_df = pd.read_pickle(PATH_HOUSEKEEPING)
feat = house_df.columns
house_df = house_df.sort_values(by='TimeJD')
df_merged = pd.merge_asof(house_df, df, left_on='TimeJD', right_on='CLARA_OLR_time', direction = 'nearest')
df_merged = df_merged.drop('CLARA_OLR_time', axis=1)
mask = (df_merged['CLARA_solar_zenith_angle'] > angle)
df_merged = df_merged[mask]
if histogram:
    tmp = df_merged['CLARA_radiance']
    mask1 = tmp < 1000
    mask2 = tmp > -100
    plt.figure(figsize=(8, 4))
    plt.hist(tmp[mask1 & mask2].dropna(), bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('CLARA_radiance')
    plt.ylabel('Frequency')
    plt.title('Histogram of CLARA radiance')
    plt.tight_layout()
    plt.savefig(PATH_HIST)
df_merged.to_pickle(PATH_OUT)
t1 = t.time()

time_elapsed = t1 - t0

print("Execution time: ", int(time_elapsed / 60), " Minutes ", int(time_elapsed % 60)," Seconds.")