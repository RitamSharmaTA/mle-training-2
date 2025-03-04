import argparse
import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor

from housepred.logger import setup_logging

logger = logging.getLogger("script")


def train_model(input_path, output_path, model_type):
    """
    Trains a machine learning model using a dataset and saves the trained model.

    Parameters
    ----------
    input_path : str
        The file path to the dataset used for training the model.
    output_path : str
        The file path where the trained model will be saved.
    model_type : str
        The type of model to train. Options are: "linear", "tree", "forest".
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading data...")
    housing = pd.read_csv(input_path)
    mlflow.log_param("dataset_size", len(housing))

    # Stratified split
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]

    # Data preparation
    housing = strat_train_set.drop(["median_house_value", "income_cat"], axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    housing_tr = pd.DataFrame(
        imputer.fit_transform(housing_num),
        columns=housing_num.columns,
        index=housing.index,  # Ensure the index is retained
    )
    housing_cat = pd.get_dummies(housing[["ocean_proximity"]], drop_first=True)
    housing_prepared = housing_tr.join(housing_cat)

    # Model selection
    models = {
        "linear": LinearRegression(),
        "tree": DecisionTreeRegressor(random_state=42),
        "forest": RandomForestRegressor(random_state=42),
    }

    if model_type not in models:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")

    model = models[model_type]
    logger.info(f"Training {model_type} model...")
    mlflow.log_param("model_type", model_type)
    model.fit(housing_prepared, housing_labels)

    # Evaluate and log metrics
    train_rmse = np.sqrt(
        mean_squared_error(housing_labels, model.predict(housing_prepared))
    )
    mlflow.log_metric("train_rmse", train_rmse)
    logger.info(f"Training RMSE: {train_rmse:.4f}")

    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    mlflow.log_artifact(output_path)
    logger.info(f"Model saved to {output_path}")


def cli():
    """Command-line interface for training a model."""
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument(
        "--input-path", type=str, required=True, help="Input dataset path"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Output model path"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["linear", "tree", "forest"],
        required=True,
        help="Model type",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    parser.add_argument("--log-path", type=str, help="Log file path")
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable console logging",
    )
    args = parser.parse_args()

    setup_logging(
        log_level=args.log_level,
        log_path=args.log_path,
        no_console_log=args.no_console_log,
    )

    train_model(args.input_path, args.output_path, args.model_type)


if __name__ == "__main__":
    cli()
