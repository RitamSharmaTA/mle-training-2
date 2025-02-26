import os
import tempfile

import joblib
import mlflow
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from housepred.train import train_model


def create_dummy_data(file_path):
    data = {
        "longitude": [-122.23, -122.25, -122.26],
        "latitude": [37.88, 37.86, 37.85],
        "housing_median_age": [41, 21, 52],
        "total_rooms": [880, 7099, 1467],
        "total_bedrooms": [129, 1106, 190],
        "population": [322, 2401, 496],
        "households": [126, 1138, 177],
        "median_income": [8.3252, 8.3014, 7.2574],
        "median_house_value": [452600, 358500, 352100],
        "ocean_proximity": ["NEAR BAY", "INLAND", "NEAR OCEAN"],
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


@pytest.mark.parametrize("model_type", ["linear", "tree", "forest"])
def test_train_model(model_type):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "dummy_data.csv")
        output_path = os.path.join(tmpdir, "model.pkl")

        create_dummy_data(input_path)

        with mlflow.start_run():
            train_model(input_path, output_path, model_type)

        assert os.path.exists(
            output_path
        ), f"Model file was not created for {model_type}"
        model = joblib.load(output_path)
        assert model is not None, "Loaded model is None"

        if model_type == "linear":
            assert isinstance(model, LinearRegression)
        elif model_type == "tree":
            assert isinstance(model, DecisionTreeRegressor)
        elif model_type == "forest":
            assert isinstance(model, RandomForestRegressor)
