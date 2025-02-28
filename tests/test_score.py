import os
import sys
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from housepred.score import preprocess_features, score_model

sys.path.append(".")


@pytest.fixture
def sample_data():
    """Create a sample housing dataset for testing."""
    n_samples = 100

    np.random.seed(42)
    data = {
        "median_income": np.random.uniform(0, 10, n_samples),
        "total_rooms": np.random.randint(1, 20, n_samples),
        "total_bedrooms": np.random.randint(1, 10, n_samples),
        "households": np.random.randint(1, 5, n_samples),
        "population": np.random.randint(1, 10, n_samples),
        "median_house_value": np.random.randint(100000, 500000, n_samples),
        "ocean_proximity": np.random.choice(
            ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"],
            n_samples,
        ),
    }

    data["median_income"][0:5] = [1.0, 2.0, 3.5, 5.0, 7.0]

    return pd.DataFrame(data)


@pytest.fixture
def trained_model():
    """Create a trained model that matches the feature set in score_model."""
    features = [
        "median_income",
        "total_rooms",
        "total_bedrooms",
        "households",
        "population",
        "rooms_per_household",
        "bedrooms_per_room",
        "population_per_household",
        "ocean_proximity_NEAR BAY",
        "ocean_proximity_NEAR OCEAN",
    ]
    model = RandomForestRegressor(n_estimators=2, max_depth=2, random_state=42)

    X = np.random.rand(20, len(features))
    y = np.random.rand(20)

    model.fit(X, y)
    return model


def test_preprocess_features():
    """Test the preprocess_features function."""

    test_data = pd.DataFrame(
        {
            "total_rooms": [10, 20, 30],
            "households": [2, 4, 6],
            "total_bedrooms": [4, 8, 12],
            "population": [6, 12, 18],
        }
    )

    processed_data = preprocess_features(test_data)

    assert "rooms_per_household" in processed_data.columns
    assert "bedrooms_per_room" in processed_data.columns
    assert "population_per_household" in processed_data.columns

    assert processed_data["rooms_per_household"].iloc[0] == 5.0
    assert processed_data["bedrooms_per_room"].iloc[0] == 0.4
    assert processed_data["population_per_household"].iloc[0] == 3.0


@patch("mlflow.log_metric")
@patch("mlflow.log_artifact")
def test_score_model(
    mock_log_artifact, mock_log_metric, sample_data, trained_model, tmp_path
):
    """Test the score_model function with mocked dependencies."""

    data_path = os.path.join(tmp_path, "housing.csv")
    model_path = os.path.join(tmp_path, "model.joblib")
    output_path = os.path.join(tmp_path, "results.txt")

    sample_data.to_csv(data_path, index=False)
    joblib.dump(trained_model, model_path)

    with patch("joblib.load", return_value=trained_model):

        with patch.object(
            trained_model, "predict", return_value=np.array([200000] * 20)
        ):
            score_model(model_path, data_path, output_path)

    assert os.path.exists(output_path)

    mock_log_metric.assert_called()
    mock_log_artifact.assert_called_with(output_path)

    with open(output_path, "r") as f:
        content = f.read()
        assert "Root Mean Squared Error:" in content


def test_cli_arguments():
    """Test the CLI argument parsing."""
    with patch("argparse.ArgumentParser.parse_args") as mock_args:

        mock_args.return_value = MagicMock(
            model_path="model.joblib",
            data_path="data.csv",
            output_path="results.txt",
            log_level="INFO",
            log_path=None,
            no_console_log=False,
        )

        with patch("housepred.score.score_model") as mock_score:
            from housepred.score import cli

            cli()

            mock_score.assert_called_once_with(
                "model.joblib", "data.csv", "results.txt"
            )
