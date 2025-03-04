import argparse
import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit

from housepred.logger import setup_logging

logger = logging.getLogger("script")


def preprocess_features(data, imputer=None):
    """
    Preprocesses features for both training and test datasets,
    including feature engineering and imputation of missing values.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataset containing features for preprocessing.
    imputer : sklearn.impute.SimpleImputer, optional
        An imputer object to handle missing values. If None, no imputation is performed.
        Default is None.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the preprocessed numeric features.
    """
    logger = logging.getLogger(__name__)
    logger.info("Performing feature engineering.")

    data["rooms_per_household"] = data["total_rooms"] / data["households"]
    data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
    data["population_per_household"] = data["population"] / data["households"]

    housing_num = data.select_dtypes(include=[np.number])

    if imputer:
        housing_num = imputer.transform(housing_num)
        logger.info("Applied imputation to missing values.")

    housing_num = pd.DataFrame(
        housing_num,
        columns=data.select_dtypes(include=[np.number]).columns,
        index=data.index,
    )

    return housing_num


def score_model(model_path, data_path, output_path):
    """
    Loads a trained model, prepares test data, evaluates the model on the test set,
    and saves the evaluation results to a file.

    Parameters
    ----------
    model_path : str
        The file path where the trained model is stored.
    data_path : str
        The file path where the dataset is stored.
    output_path : str
        The file path where the results will be saved.

    Returns
    -------
    None
        Saves the RMSE results to the output file.
    """
    logger = logging.getLogger(__name__)

    logger.info("Loading model...")
    model = joblib.load(model_path)

    logger.info("Loading data...")
    housing = pd.read_csv(data_path)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Data preparation
    housing = strat_test_set.drop(["median_house_value", "income_cat"], axis=1)

    housing_labels = strat_test_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    logger.info("Scoring model...")
    predictions = model.predict(housing_prepared)
    mse = mean_squared_error(housing_labels, predictions)
    rmse = np.sqrt(mse)
    logger.info(f"Root Mean Squared Error: {rmse}")

    mlflow.log_metric("test_rmse", rmse)
    r2 = r2_score(housing_labels, predictions)
    mae = mean_absolute_error(housing_labels, predictions)

    mlflow.log_metric("test_r2", r2)
    mlflow.log_metric("test_mae", mae)

    logger.info("Saving results...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"Root Mean Squared Error: {rmse}\n")

    logger.info("Scoring completed.")
    mlflow.log_artifact(output_path)


def cli():
    """
    Command-line interface for scoring a model.

    Parses command-line arguments, sets up logging, and calls the score_model function
    to evaluate a model on a test dataset.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Score the model.")
    parser.add_argument("--model-path", type=str, required=True, help="Model path")
    parser.add_argument("--data-path", type=str, required=True, help="Dataset path")
    parser.add_argument(
        "--output-path", type=str, required=True, help="Output results path"
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

    score_model(args.model_path, args.data_path, args.output_path)


if __name__ == "__main__":
    cli()
