import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random
import math
from utils import *
import time as t

## HYPERPARAMETER ################################################################################

t0 = t.time()

# Read-in:
DARA = False

PATH_DATA = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/combined_data.pkl' if DARA else '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/CLARA_combined_all.pkl'
PATH_FEATURES = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/combined_data.pkl' if DARA else '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/CLARA_combined_all.pkl'
TARGET_PATH = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/' if DARA else '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/CLARA_'
IMAGE_PATH = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Images/' if DARA else '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Images/CLARA_'

# Preprocessing:
OUTLIER_UPPER = 1370 if DARA else 600
OUTLIER_LOWER = 1356 if DARA else -100
THRESHOLD = 3 if DARA else 238 
# only use if combined_data includes at least 2021-2023 data and want to reproduce the DARA plot from the DS-lab paper
paper_reproduction = False 

################################################################################################

## READ-IN #####################################################################################

df = pd.read_pickle(PATH_DATA)
feature_list = read_pickle(PATH_FEATURES)

missing_features = [feature for feature in feature_list if feature not in df.columns]
if missing_features:
    print("Warning: The following features are not present in the DataFrame:", missing_features)

columns = [i for i in df.columns if i in feature_list]
df = df[columns]

print("READ-IN: complete.")

## PREPROCESSING #################################################################################

df = df.sort_values(by=['TimeJD'])

# Safety tests and data validation checks

missing_values = df.isnull().sum()
if missing_values.any():
    print("Warning: Missing values found. Handling missing values if not in IrrB...")
    if DARA:
        columns = df.columns[df.columns != 'IrrB']
    else:
        columns = df.columns[df.columns != 'CLARA_radiance']
    # Deleting rows with missing values (do not consider IrrB in light of gap filling)
    df = df.dropna(subset=columns)
    # Creating a fully clean dataset as well
    df_clean = df.dropna()
else:
    df_clean = df.copy()
    print("SANITY CHECK: No missing values found.")

# Range Validation
# Assuming IrrB should be within [OUTLIER_LOWER, OUTLIER_UPPER] range
if DARA:
    out_of_range_indices = df[(df['IrrB'] < OUTLIER_LOWER) | (df['IrrB'] > OUTLIER_UPPER)].index
else: 
    out_of_range_indices = df[(df['CLARA_radiance'] < OUTLIER_LOWER) | (df['CLARA_radiance'] > OUTLIER_UPPER)].index
if not out_of_range_indices.empty:
    print("Warning: Values outside expected range found. Handling out-of-range values...")
    # Handling out-of-range values
    if DARA:
        df = df[((df['IrrB']<OUTLIER_UPPER) & (df['IrrB']>OUTLIER_LOWER)) | df['IrrB'].isna()]
    else:
        df = df[((df['CLARA_radiance']<OUTLIER_UPPER) & (df['CLARA_radiance']>OUTLIER_LOWER)) | df['CLARA_radiance'].isna()]

# Unique Values
duplicates = df.duplicated()
if duplicates.any():
    print("Warning: Duplicates found. Removing duplicates.")
    df = df.drop_duplicates()
    
################################################################################################

## CORRELATION PLOT ############################################################################

# Generate data and set axis
data = df_clean.corr(numeric_only=True)
f, ax = plt.subplots(figsize=(10, 8))

# Define colormap and generate correlation plot
cmap = plt.cm.coolwarm
image = ax.matshow(data, cmap=cmap)

# Add colorbar, set ticks, labels and the title
cb = plt.colorbar(image)
ax.set_xticks(np.arange(data.shape[1]))
ax.set_yticks(np.arange(data.shape[0]))
ax.set_xticklabels(data.columns, rotation=45, fontsize=8, ha='left')
ax.set_yticklabels(data.columns, fontsize=8)
plt.title('Correlation Matrix for CLARA data', fontsize=20)

# Add text annotations to see individual correlation values
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, "{:.2f}".format(data.iloc[i, j]), ha='center', va='center', color='black', fontsize=6)

# Save plot in the desired folder
plt.savefig(IMAGE_PATH + 'correlation_plot.png')

################################################################################################

## OUTLIER REMOVAL #############################################################################

# Checking for Outliers and removing them with rolling median
df.reset_index(inplace=True)

# Set the size of the rolling window for calculating the median and compute median and deviation
window_size = 5 if DARA else 50
if DARA:
    rolling_median = df['IrrB'].dropna().rolling(window=window_size).median()
    deviation = abs(df['IrrB'].dropna() - rolling_median)
else: 
    rolling_median = df['CLARA_radiance'].dropna().rolling(window=window_size).median()
    deviation = abs(df['CLARA_radiance'].dropna() - rolling_median)
# Eliminate outliers beyond threshold
outliers_mask = deviation >= THRESHOLD
if DARA:
    original_length = len(df["IrrB"])
    outlier_mask_df = outliers_mask.to_frame().reset_index()
    outlier_indx = outlier_mask_df[outlier_mask_df['IrrB']]['index']
    noutlier_indx = outlier_mask_df[~outlier_mask_df['IrrB']]['index']
else:
    original_length = len(df["CLARA_radiance"])
    outlier_mask_df = outliers_mask.to_frame().reset_index()
    outlier_indx = outlier_mask_df[outlier_mask_df['CLARA_radiance']]['index']
    noutlier_indx = outlier_mask_df[~outlier_mask_df['CLARA_radiance']]['index']

# Create a new DataFrame containing only the outliers
outliers = df.drop(noutlier_indx, axis=0)
if len(outliers) > 0:
    print("Warning: Outliers found. Handling outliers...")

# Remove the outliers from the original DataFrame
df = df.drop(outlier_indx, axis=0)
if DARA:   
    print(f'Removing: {original_length-len(df["IrrB"])} datapoints')
else:
    print(f'Removing: {original_length-len(df["CLARA_radiance"])} datapoints')

# Create plots for visualizing the original data and outliers
fig, axes = plt.subplots(2,1, figsize=(15, 8))
if DARA:
    sns.scatterplot(x=df["TimeJD"], y=df["CLARA_radiance"], ax = axes[0]).set(title=f'Scatterplot of IrrB Median - threshold: {THRESHOLD}')
    sns.scatterplot(x=outliers['TimeJD'], y=outliers['CLARA_radiance'], ax = axes[1], color='red')
else:
    sns.scatterplot(x=df["TimeJD"], y=df["CLARA_radiance"], ax = axes[0]).set(title=f'Scatterplot of IrrB Median - threshold: {THRESHOLD}')
    sns.scatterplot(x=outliers['TimeJD'], y=outliers['CLARA_radiance'], ax = axes[1], color='red')

# Save plot in the desired folder
plt.savefig(IMAGE_PATH + 'outlier_plot.png')

################################################################################################

if paper_reproduction:
    end = pd.Timestamp('2023-09-01')

    df = df[(df['TimeJD'] < end)]

## GAP FINDING #################################################################################

# Separate the data in train and test based on missing values
if DARA:
    df_train = df[df['IrrB'].notna()].copy()
    df_test = df[df['IrrB'].isna()].copy()
else:
    df_train = df[df['CLARA_radiance'].notna()].copy()
    df_test = df[df['CLARA_radiance'].isna()].copy()

# Set minimum treshold to fill and find gaps bigger than that
gap_threshold = pd.Timedelta(days=1)
df_train['gap'] = df_train['TimeJD'].diff()
large_gaps_mask = (df_train['gap'] > gap_threshold)
selected_gaps = df_train[large_gaps_mask]

# Resample all data and test data
time_interval = pd.to_timedelta('15 minutes')
df = df.drop('index', axis = 1)
df.set_index('TimeJD', inplace=True)
df_test = df_test.drop('index',axis = 1)
df_test.set_index('TimeJD', inplace=True)
df_resampled = df.resample(time_interval).mean().copy()

# Drop NAs due to the averaging
if DARA:
    df_resampled.dropna(subset=df_test.columns.difference(['IrrB']), inplace=True)
    df_test = df_resampled[df_resampled['IrrB'].isna()].copy()
else:
    df_resampled.dropna(subset=df_test.columns.difference(['CLARA_radiance']), inplace=True)
    df_test = df_resampled[df_resampled['CLARA_radiance'].isna()].copy()

# Select the desired rows
sampled_rows = pd.DataFrame()
for index, gap in selected_gaps.iterrows():
    end_time = gap['TimeJD']
    start_time = end_time - gap['gap']

    current_rows = df_test.loc[start_time:end_time].copy()
    sampled_rows = pd.concat([sampled_rows, current_rows])

# Reset the index for the final result
sampled_rows.reset_index(inplace=True)
df_train.drop('gap', axis = 1, inplace = True)
df_train.drop('index', axis = 1, inplace = True)

df_resampled = df_resampled.reset_index()
df_resampled['TimeJD'] = pd.to_datetime(df_resampled['TimeJD'])
df_test = df_test.reset_index()
df_test['TimeJD']= pd.to_datetime(df_test['TimeJD'])

################################################################################################

## SAVE PREPROCESSED DATA ######################################################################

df_train.to_pickle(TARGET_PATH + 'df_train.pkl')
df_test.to_pickle(TARGET_PATH + 'df_test.pkl')

t1 = t.time()
time_elapsed = t1 - t0

print("Execution time: ", int(time_elapsed / 60), " Minutes ", int(time_elapsed % 60)," Seconds.")

################################################################################################