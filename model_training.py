import os
import mlflow
import dagshub
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

def data_extraction(mongo_uri):
    client = MongoClient(mongo_uri)
    db = client["aqi_data"]
    collection = db["karachi_aqi_etl"]
    df = pd.DataFrame(list(collection.find()))
    return df

def data_preprocessing(df):

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["dayofweek"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month

    df["aqi_change_rate"] = df["aqi"].diff()

    # Multi-step forecasting
    df["aqi_t+1"] = df["aqi"].shift(-24)
    df["aqi_t+2"] = df["aqi"].shift(-48)
    df["aqi_t+3"] = df["aqi"].shift(-72)

    df.dropna(inplace=True)

    return df

def data_splitting(df):

    features = [
        "co", "no2", "o3", "pm10", "pm2_5", "so2",
        "hour", "day", "dayofweek", "month", "aqi_change_rate"
    ]

    X = df[features]
    y = df[["aqi_t+1", "aqi_t+2", "aqi_t+3"]]

    split = int(0.8 * len(df))

    return (
        X[:split], X[split:],
        y[:split], y[split:]
    )

def train_random_forest(X_train, y_train):
    model = MultiOutputRegressor(
        RandomForestRegressor(random_state=42)
    )

    param_grid = {
        "estimator__n_estimators": [100, 200],
        "estimator__max_depth": [None, 10, 20]
    }

    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    return grid.best_estimator_, param_grid


def train_xgboost(X_train, y_train):
    model = MultiOutputRegressor(
        xgb.XGBRegressor(random_state=42)
    )

    param_grid = {
        "estimator__n_estimators": [100, 200],
        "estimator__learning_rate": [0.05, 0.1],
        "estimator__max_depth": [4, 6]
    }

    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    return grid.best_estimator_, param_grid


def train_svr(X_train, y_train):
    model = MultiOutputRegressor(SVR())

    param_grid = {
        "estimator__kernel": ["rbf", "linear"],
        "estimator__C": [0.1, 1]
    }

    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    return grid.best_estimator_, param_grid

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, mae, r2

def register_model(model, mse, mae, r2, param_grid):

    """
    Register model to DagsHub MLflow using token authentication
    """
    
    # Get the DagsHub token from environment
    dagshub_token = os.getenv("DAGSHUB_REPO_TOKEN")
    
    if not dagshub_token:
        raise ValueError("DAGSHUB_REPO_TOKEN environment variable is not set")
    
    # Set MLflow tracking URI with authentication
    mlflow_tracking_uri = "https://dagshub.com/hasnainhissam56/AQI_Predictor_Models.mlflow"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Set authentication using environment variables (MLflow will use these)
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    with mlflow.start_run() as run:

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="AQI_Predictor_Model"
        )

        mlflow.log_metrics({
            "mse": mse,
            "mae": mae,
            "r2": r2
        })

        mlflow.log_params(param_grid)

        print("Model Registered Successfully")
        return run.info.run_id


# ================================
# 7. MAIN PIPELINE
# ================================

def main():

    MONGO_URI = os.getenv("MONGO_URI")

    df = data_extraction(MONGO_URI)
    df = data_preprocessing(df)
    X_train, X_test, y_train, y_test = data_splitting(df)

    models = {}

    # Train all models
    for name, trainer in {
        "XGBoost": train_xgboost,
        "RandomForest": train_random_forest,
        "SVR": train_svr
    }.items():

        model, params = trainer(X_train, y_train)
        mse, mae, r2 = evaluate_model(model, X_test, y_test)

        models[name] = {
            "model": model,
            "params": params,
            "mse": mse,
            "mae": mae,
            "r2": r2
        }

        print(f"{name} → R2: {r2}")

    # Select Best Model
    best_model_name = max(models, key=lambda x: models[x]["r2"])
    best = models[best_model_name]

    print(f"\nBest Model: {best_model_name}")

    # Register Best Model
    register_model(
        best["model"],
        best["mse"],
        best["mae"],
        best["r2"],
        best["params"]
    )


if __name__ == "__main__":
    main()
