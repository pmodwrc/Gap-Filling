# TSI-Prediction

This project focuses on AI-based gap filling of missing irradiance data for the two radiometers CLARA and DARA. The implemented models include unidirectional and bidirectioal LSTMs, neural networks (NN), temporal convolutional networks (TCN), and time series transfomers (PatchTST). 

## Project Overview

This project provides tools for:
- Preprocessing the radiometer data
- Imputing missing features and targets
- Training and evaluating machine learning models (LSTM, BILSTM, TCN, PatchTST, NN)
- Gap filling and forecasting of missing irradiance values

## Installation

To install the necessary packages use:

**For Conda environments:**
```sh
conda env create -f environment.yml
```

**For pip installation:**
```sh
pip install -r requirements.txt
```

## Usage

**For DARA:**

1. Preprocess the data:
    First the Level1A and Level2A fits file need to be read in and merged. Required inputs are Level1A and Level2A fits files in the correct folder structure (see below). Output is a pickle file with all the data.
    - In the file 'read_combine.py' set:
        - DARA = True
        - SOURCE_L1_PATH, SOURCE_L2_PATH, PATH_FEATURES, TARGET_PATH to the desired path
    - Make sure to:
        - Create a Data folder if it does not already exist
        - Respect the folder structure described in the function description of 'list_files_in_subfolders'
    - Run the file: 
        ```sh
        python read_combine.py
        ```
    Afterwards the merged file can be split into train and test datasets for model training. Input is the previously constructed pickle file and output are 2 pickle files for the training and test data. 
    - In the file 'preprocessing.py' set:
        - DARA = True
        - PATH_DATA, PATH_FEATURES, TARGET_PATH, IMAGE_PATH to the desired path
    - Run the file: 
        ```sh
        python preprocess.py
        ```
    To fill the gaps in the data the test dataset needs to be imputed. Input is the previously constructed test data pickle file and output is a new imputed test pickle file.
    - In the file 'postprocessing.py' set:
        - Dara = True
        - PATH_TRAIN, PATH_TEST, TARGET_PATH to the desired path
        - mode to the desired imputation mode
        - amount to 'first' to use the same amount of timepoints for all gap predictions, 'day before' to predict at the same hour and minute points as the day before the gap
    - Run the file: 
        ```sh
        python postprocess.py
        ```
2. Train the model:
    Inputs are the train and test pickle files and outputs are a trained model, two plots of the models' predictions and a plot of the train loss. 
    - In the file 'models.py' set:
        - DARA = True
        - PATH_TRAIN, PATH_TEST, TARGET_PATH to the desired path
        - SPLIT = 0 to train in the entire train dataset (no validation data) and SPLIT > 0 to artificially produce gaps for validation and compute prediction losses
        - time_features = True to include year, month, day, hour and minute as features for model training
        - hyperparameters hidden_size, learning_rate, num_epochs, dropout, num_layers, window to the desired values (see report for suitable values)
        - gap_filling = True to train the model for a gap filling task otherwise it will be trained for forecasting
        - lstm = True and bidirectional = False to train an unidirectional LSTM
        - lstm = True and bidirectional = True to train an bidirectional LSTM
        - lstm = False and ff = True to train a neural network
        - lstm = False, ff = False and tcn = True to train a temporal convolutional network
        - lstm = False, ff = False and tcn = False to train a time series transformer
        - gap = -1 and months = -1 to use the entire dataset for training and artificially include 6 gaps with length 1-6 days (requires SPLIT > 0)
        - gap > 0 to artificially include 1 gap of 'gap' days (requires SPLIT > 0)
        - months > 0 to use 'months' months for training
        - train_loss = True to compute the training loss on the unscaled data (recommended)
        - plot_train = True to plot the predictions on the training set too
    - Run the file: 
        ```sh
        python models.py
        ```
3. Make predictions:
    To use a previously trained model to make predictions on another gap or reconstruct results without having to retrain the model. Inputs are the train and test pickle files as well as the previously trained model and outputs are the models' predictions.
    - In the file 'predict.py' set:
        - DARA = True
        - PATH_TRAIN, PATH_TEST, TARGET_PATH and MODEL_PATH to the desired path
        - The hyperparameters to the same values used in training
        - Variables which can be changed are: SPLIT, gap, window, train_loss, plot_train, impute, mode
    - Run the file: 
        ```sh
        python predict.py
        ```

**For CLARA:**

1. Preprocess the data:
    First the fits files need to be read in and merged. Required inputs are Level1A fits files in the correct folder structure (see below). Output is a pickle file with all the data.
    - In the file 'read_combine.py' set:
        - DARA = False
        - SOURCE_L1_PATH, SOURCE_L2_PATH, PATH_FEATURES, TARGET_PATH to the desired path
    - Make sure to:
        - Create a Data folder if it does not already exist
        - Respect the folder structure described in the function description of 'list_files_in_subfolders'
    - Run the file: 
        ```sh
        python read_combine.py
        ```
    Now the housekeeping features can be combined with the OLR and position features. Required inputs are the previously created CLARA_combined_data.pkl file, all OLR files (.save) in a single directory, and all position files (.pkl) in a single directory. Outputs are a CLARA_OLR_combined.pkl file, a CLARA_combined_all.pkl file (contains housekeeping, position and OLR data) and a histogram for the CLARA radiance.
    - In the file 'transform_CLARA_data.py' set:
        - DARA = False
        - PATH_OLR, PATH_POS, PATH_HOUSEKEEPING, PATH_TARGET, PATH_OUT, PATH_HIST to the desired path
    - Make sure to:
        - Create a Data folder if it does not already exist
        - Respect the folder structure described in the function description of 'list_files_in_subfolders'
                - angle (for CLARA_solar_zenith_angle) as desired
        - histogram = True to plot the histogram of the CLARA radiation
        - Run the file: 
        ```sh
        python read_combine.py
        ```
    Afterwards the merged file can be split into train and test datasets for model training. Input is the previously constructed pickle file and output are 2 pickle files for the training and test data. 
    - In the file 'preprocessing.py' set:
        - DARA = False
        - PATH_DATA, PATH_FEATURES, TARGET_PATH, IMAGE_PATH to the desired path
    - Run the file: 
        ```sh
        python preprocess.py
        ```
    To fill the gaps in the data the test dataset needs to be imputed. Input is the previously constructed test data pickle file and output is a new imputed test pickle file.
    - In the file 'postprocessing.py' set:
        - Dara = False
        - PATH_TRAIN, PATH_TEST, TARGET_PATH to the desired path
        - mode to the desired imputation mode
        - amount to 'first' to use the same amount of timepoints for all gap predictions, 'day before' to predict at the same hour and minute points as the day before the gap
    - Run the file: 
        ```sh
        python postprocess.py
        ```
2. Train the model:
    Inputs are the train and test pickle files and outputs are a trained model, two plots of the models' predictions and a plot of the train loss. 
    - In the file 'models.py' set:
        - DARA = False
        - PATH_TRAIN, PATH_TEST, TARGET_PATH to the desired path
        - SPLIT = 0 to train in the entire train dataset (no validation data) and SPLIT > 0 to artificially produce gaps for validation and compute prediction losses
        - time_features = True to include year, month, day, hour and minute as features for model training
        - hyperparameters hidden_size, learning_rate, num_epochs, dropout, num_layers, window to the desired values (see report for suitable values)
        - gap_filling = True to train the model for a gap filling task otherwise it will be trained for forecasting
        - lstm = True and bidirectional = False to train an unidirectional LSTM
        - lstm = True and bidirectional = True to train an bidirectional LSTM
        - lstm = False and ff = True to train a neural network
        - lstm = False, ff = False and tcn = True to train a temporal convolutional network
        - lstm = False, ff = False and tcn = False to train a time series transformer
        - gap = -1 and months = -1 to use the entire dataset for training and artificially include 6 gaps with length 1-6 days (requires SPLIT > 0)
        - gap > 0 to artificially include 1 gap of 'gap' days (requires SPLIT > 0)
        - months > 0 to use 'months' months for training
        - train_loss = True to compute the training loss on the unscaled data (recommended)
        - plot_train = True to plot the predictions on the training set too
    - Run the file: 
        ```sh
        python models.py
        ```
3. Make predictions:
    To use a previously trained model to make predictions on another gap or reconstruct results without having to retrain the model. Inputs are the train and test pickle files as well as the previously trained model and outputs are the models' predictions.
    - In the file 'predict.py' set:
        - DARA = False
        - PATH_TRAIN, PATH_TEST, TARGET_PATH and MODEL_PATH to the desired path
        - The hyperparameters to the same values used in training
        - Variables which can be changed are: SPLIT, gap, window, train_loss, plot_train, impute, mode
    - Run the file: 
        ```sh
        python predict.py
        ```

## ETH cluster setup

1. Copy the content of the directory Euler to the cluster
2. Include the respective train and test pickle files in the Data folder
3. The number of Job directories (models trained at the same time) can be increased up to 8
4. Check out slurm documentation to submit, cancel or get information about jobs