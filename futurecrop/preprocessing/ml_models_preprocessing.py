import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def get_annual_data(table: dict):
    """
    this function computes annual values for the features:
        - total volume of precipitation
        - means of tas, tasmin, tasmax and rsds
        - and concats them with the soil datas
        - returns a dataframe
    we need this to feed ML models
    """

    # computes what needs to be computed
    annual_pr = table['pr'].iloc[:,5:].sum(axis=1)
    annual_tas = table['tas'].iloc[:,5:].mean(axis=1)
    annual_tasmin = table['tasmin'].iloc[:,5:].mean(axis=1)
    annual_tasmax = table['tasmax'].iloc[:,5:].mean(axis=1)
    annual_rsds = table['rsds'].iloc[:,5:].mean(axis=1)

    # concats all the annual datas in a dataframe
    if 'target' in table:
        annual_data = pd.concat([table['pr'][['year', 'lon', 'lat', 'crop']], # keep the identifications values ?
                                annual_pr,
                                annual_tas,
                                annual_tasmin,
                                annual_tasmax,
                                annual_rsds,
                                table['soil_co2'][['texture_class', 'co2', 'nitrogen']],
                                table['target']],
                                axis=1)

        annual_data.columns = ['year',
                            'lon',
                            'lat',
                            'crop',
                            'total_precipitation',
                            'mean_tas',
                            'mean_tasmin',
                            'mean_tasmax',
                            'mean_rsds',
                            'texture_class',
                            'co2',
                            'nitrogen',
                            'yield']
    else:
        annual_data = pd.concat([table['pr'][['year', 'lon', 'lat', 'crop']], # keep the identifications values ?
                                annual_pr,
                                annual_tas,
                                annual_tasmin,
                                annual_tasmax,
                                annual_rsds,
                                table['soil_co2'][['texture_class', 'co2', 'nitrogen']]],
                                axis=1)

        annual_data.columns = ['year',
                            'lon',
                            'lat',
                            'crop',
                            'total_precipitation',
                            'mean_tas',
                            'mean_tasmin',
                            'mean_tasmax',
                            'mean_rsds',
                            'texture_class',
                            'co2',
                            'nitrogen']


    return annual_data


def val_train_split_the_data(table, test_size=0.3):
    """
    this functions splits the data in X_train, X_val, y_train, y_val
    """

    features = ['year', 'lon', 'lat',
            'crop', 'total_precipitation', 'mean_tas', 'mean_tasmin',
            'mean_tasmax', 'mean_rsds', 'texture_class', 'co2', 'nitrogen']

    X = table[features]
    y = table['yield']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

    return X_train, X_val, y_train, y_val
