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
    return r2_random_forest


def rf_prediction(model, X, y):
    predictions = model.predict(X, y)
    print("🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮​🔮")
    return predictions
