import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
from datetime import datetime
import os

from futurecrop.preprocessing.basic_prep import load_data, handle_column_year, crop_encoding
from futurecrop.preprocessing.ml_models_preprocessing import get_annual_data, val_train_split_the_data
from futurecrop.params import *


def init_random_forest_model(n_estimators=200,
                             max_depth=None):
    """
    initializes a Random Forest model
    the defaults params (n_estimators=200, max_depth=None) are the ones found after a gridsearchcv
    """
    print("Model initialized!")
    return RandomForestRegressor(n_estimators=n_estimators,
                                 max_depth=max_depth)


def train_random_forest_model(model, X, y):
    model.fit(X, y)
    print("🏋️​🏋️​ Model trained 🏋️​🏋️​")
    return model


def evaluate_random_forest_model(model, X, y):

    r2_random_forest = model.score(X, y)
    print(f"📈​ The R2 of the model Random Forest is {r2_random_forest} ​📉​")

    y_pred = model.predict(X)
    rmse_random_forest = np.sqrt(np.mean((y_pred - y)**2))
    print(f"📈​ The RMSE of the model Random Forest is {rmse_random_forest} ​📉​")
    return r2_random_forest, rmse_random_forest

def save_model(model, path='model.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"The trained model has been saved here : {path}")
    return model

def load_model(path='model.pkl'):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print("The trained model has been loaded !")
    return model

def rf_prediction(model, X, directory: str):
    """this functions makes predictions
    and saves them in a .csv file ready for submission
    """
    predictions = model.predict(X)
    print("🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮")

    # saving predictions as submissions.csv
    test_ids = X.index
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'yield': predictions
    })

    # create the submissions directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(directory, f"{timestamp}.csv")
    submission_df.to_csv(path, index=False)
    print("The submissions have been saved !")
    return predictions


if __name__ == '__main__':
    # loading and basic preproc
    maize = load_data(crop='maize', mode=MODE, path_to_data_directory=DATA_DIRECTORY)
    maize = handle_column_year(maize)
    maize = get_annual_data(maize)

    wheat = load_data(crop='wheat', mode=MODE, path_to_data_directory=DATA_DIRECTORY)
    wheat = handle_column_year(wheat)
    wheat = get_annual_data(wheat)

    annual_maize_and_wheat = pd.concat((maize, wheat))
    annual_maize_and_wheat = crop_encoding(annual_maize_and_wheat)
    print(annual_maize_and_wheat.columns)



    if MODE == 'train':
        # train_val_split
        X_train, X_val, y_train, y_val = val_train_split_the_data(annual_maize_and_wheat)

        # initialize the model
        model = init_random_forest_model()
        # train the model
        model = train_random_forest_model(model=model,
                                          X=X_train,
                                          y=y_train)
        # evaluate the model
        r2_random_forest, rmse_random_forest = evaluate_random_forest_model(model=model,
                                                                            X=X_val,
                                                                            y=y_val)
        # save the model trained
        save_model(model=model, path=PATH_TO_MODEL)


    elif MODE == 'test':
        # load the trained model
        model = load_model(path=PATH_TO_MODEL)

        # features = ['crop', 'total_precipitation', 'mean_tas', 'mean_tasmin',
        #     'mean_tasmax', 'mean_rsds', 'texture_class', 'co2', 'nitrogen']
        # annual_maize_and_wheat = annual_maize_and_wheat[features]

        # make predictions and store them as .csv
        predictions = rf_prediction(model=model,
                                    X=annual_maize_and_wheat,
                                    directory=SUBMISSION_DIRECTORY)
