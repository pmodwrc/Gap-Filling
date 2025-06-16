import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random
import math
from utils import *
from datetime import datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

## HYPERPARAMETER ################################################################################
DARA = True

PATH_TRAIN = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/df_train.pkl'
PATH_TEST = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/df_test.pkl'
TARGET_PATH = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/df_test_'
mode = 'mean'  # mean, median, knn_impute, iter_impute
amount = 'first' # first or day before

##################################################################################################

## READ-IN #####################################################################################

df_train = pd.read_pickle(PATH_TRAIN)
df_test = pd.read_pickle(PATH_TEST)

print("READ-IN: complete.")

################################################################################################

train_dates = sorted(df_train['TimeJD'].dt.date.unique())
test_dates = sorted(df_test['TimeJD'].dt.date.unique())
all_dates = sorted(train_dates + test_dates)
all_dates_np = np.array(all_dates, dtype='datetime64[D]')
# Generate the full range of dates
full_range = np.arange(all_dates_np[0], all_dates_np[-1] + np.timedelta64(1, 'D'), dtype='datetime64[D]')
missing_dates = set(full_range) - set(all_dates_np)
missing_dates = sorted(missing_dates)

# determine all gap lengths
gap_lengths = []
current_gap = []
for i, date in enumerate(missing_dates):
    if i == 0:
        current_gap = [date]
    else:
        # Check if this date is consecutive to the previous missing date
        if (date - missing_dates[i-1]) == np.timedelta64(1, 'D'):
            current_gap.append(date)
        else:
            gap_lengths.append(len(current_gap))
            current_gap = [date]
if current_gap:
    gap_lengths.append(len(current_gap))
    
gap_dict = {}
ctr = 0
for i, gap in enumerate(gap_lengths):
    tmp =[]
    for j in range(gap):
        tmp.append(missing_dates[ctr + j])
    ctr += len(tmp)
    gap_dict[f"gap{i}"] = tmp

if amount == 'first':
    times = None

for i in range(len(gap_dict)):
    current_gap = gap_dict[f'gap{i}']
    gap_start = current_gap[0]
    gap_end = current_gap[-1]
    # Find the date before and after the gap in the full date range
    idx_start = np.where(full_range == gap_start)[0][0]
    idx_end = np.where(full_range == gap_end)[0][0]

    date_before = full_range[idx_start - 1] if idx_start > 0 else None
    date_after = full_range[idx_end + 1] if idx_end + 1 < len(full_range) else None
    mask_before = df_train['TimeJD'].dt.date.isin([date_before])
    mask_after = df_train['TimeJD'].dt.date.isin([date_after])
    mask_combined = mask_before | mask_after
    # Calculate mean or median for each feature 
    if DARA: 
        feature_cols = [col for col in df_train.columns if col not in ['IrrB', 'TimeJD']]
    else:
        feature_cols = [col for col in df_train.columns if col not in ['CLARA_radiance', 'TimeJD']]
    selected_rows = df_train.loc[mask_combined, feature_cols]
    if mode == 'mean':
        feature_values = selected_rows.mean()
    elif mode == 'median':
        feature_values = selected_rows.median()
    elif mode == 'knn_impute':
        feature_values = selected_rows.fillna(np.nan)
        imputer = KNNImputer(n_neighbors=5)
    else:
        feature_values = selected_rows.fillna(np.nan)
        imputer = IterativeImputer(random_state=42, max_iter=10)
    n_points = max(len(df_train[mask_before]),len(df_train[mask_after])) #len(df_train[mask_before]) if len(df_train[mask_before]) > 0 else len(df_train[mask_after])
    mask = mask_before if len(df_train[mask_before]) == n_points else mask_after
    new_rows = []

    for gap_date in current_gap:
        if n_points > 0:
            if amount != 'first':
                times = df_train.loc[mask, 'TimeJD']
            elif amount == 'first' and i == 0:
                times = df_train.loc[mask, 'TimeJD']
            for t in times:
                row = feature_values.to_dict()
                if DARA:
                    row['IrrB'] = np.nan
                else:
                    row['CLARA_radiance'] = np.nan
                time_part = pd.to_datetime(t).time()
                row['TimeJD'] = pd.Timestamp(datetime.combine(pd.Timestamp(gap_date), time_part))
                new_rows.append(row)

    gap_filled_df = pd.DataFrame(new_rows)
    df_test_extended = pd.concat([df_test, gap_filled_df], ignore_index=True)
    df_test = df_test_extended.sort_values('TimeJD').reset_index(drop=True)

if 'impute' in mode:
    df = pd.concat([df_train, df_test], ignore_index=True)
    numeric_cols = df.columns.tolist()
    # Exclude 'IrrB' from imputation
    if DARA:
        numeric_cols = [col for col in numeric_cols if (col != 'IrrB' and col != 'TimeJD')]
    else: 
        numeric_cols = [col for col in numeric_cols if (col != 'CLARA_radiance' and col != 'TimeJD')]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    tmp = df[numeric_cols]
    imputed = imputer.fit_transform(tmp)
    df_imputed = pd.DataFrame(imputed, columns=numeric_cols, index=df.index)
    df[numeric_cols] = df_imputed
    df_test = df.iloc[len(df_train):].reset_index(drop=True)

df_test.to_pickle(TARGET_PATH + f'postprocessed_{mode}.pkl')