import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor



def init_random_forest_model(n_estimators=200,
                             max_depth=None):
    """
    initializes a Random Forest model
    the defaults params (n_estimators=200, max_depth=None) are the ones found after a gridsearchcv
    """
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


def rf_prediction(model, X, path: str):
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
    submission_df.to_csv(path, index=False)
    print("The submissions have been saved !")
    return predictions
