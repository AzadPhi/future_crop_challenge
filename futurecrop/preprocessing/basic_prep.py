import os
import numpy as np
import pandas as pd


def load_data(crop: str, mode: str='train', path_to_data_directory=None):
    """
    loads the tables from the data folder
    returns a dict with 6 dataframes : tas, tasmin, tasmax, rsds, pr, soil_co2
    if mode='train', adds the target table (yields) to the 6 dataframes
    if no path to raw_data_directory is given, it assumes the relative path to the directory is "data/the-future-crop-challenge"
    """

    # if no path to data is given, takes the relative path
    if path_to_data_directory == None:
        path_to_data_directory = os.path.join('data', 'the-future-crop-challenge')


    # dictionary with features dataframes
    features_list = ['tas', 'tasmin', 'tasmax', 'rsds', 'pr', 'soil_co2']
    data = {}
    for feat in features_list:
        data[feat] = pd.read_parquet(
            os.path.join(path_to_data_directory, f"{feat}_{crop}_{mode}.parquet")
            )

    # includes the yields df for the trains
    if mode != 'test':
        data['target'] = pd.read_parquet(
            os.path.join(path_to_data_directory, f"{mode}_solutions_{crop}.parquet")
            )
    print(f"The table {crop}_{mode} has been loaded as a dict !")
    print(f"It contains the following dataframes: {data.keys()}")
    return data




def handle_column_year(data: dict):
    """
    adds 1601 to the year column to get the "real" year
    in case there are two columns 'year' and 'real_year': keeps only the 'real_year' column and renames it
    """

    for key in data.keys():
        if 'real_year' not in data[key].columns and 'year' in data[key].columns:
            data[key]['year'] += 1601
        elif 'real_year' in data[key].columns:
            data[key] = data[key].drop(columns='year')
            data[key] = data[key].rename(columns={'real_year': 'year'})

    print("The column year has been processed!")
    return data

def crop_encoding(X):
    """
    simple encoding for the crop column : {'maize': 1, 'wheat': 0}
    """
    X['crop'] = X['crop'].map({'maize': 1, 'wheat': 0})
    print("The column 'crop' has been encoded!")
    return X
