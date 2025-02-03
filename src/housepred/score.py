import argparse
import logging
import os
import mlflow
import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor


def preprocess_features(data, imputer=None):
    """
    Preprocesses features for both training and test datasets, including feature engineering
    and imputation of missing values.

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
    # Feature engineering
    data["rooms_per_household"] = data["total_rooms"] / data["households"]
    data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
    data["population_per_household"] = data["population"] / data["households"]

    # Select numeric features and impute missing values
    housing_num = data.select_dtypes(include=[np.number])
    if imputer:
        housing_num = imputer.transform(housing_num)
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
        This function does not return anything; it saves the RMSE results to the output file.
    """
    logging.basicConfig(level=logging.INFO)
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
    for train_index, test_index in split.split(
        housing, housing["income_cat"]
    ):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Data preparation
    housing = strat_test_set.drop("median_house_value", axis=1)
    housing_labels = strat_test_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )

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
        This function does not return anything. It handles the command-line interface and
        invokes scoring logic.
    """
    parser = argparse.ArgumentParser(description="Score the model.")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Model path"
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Dataset path"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Output results path"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Log level"
    )
    parser.add_argument("--log-path", type=str, help="Log file path")
    parser.add_argument(
        "--no-console-log", action="store_true", help="Toggle console logging"
    )
    args = parser.parse_args()

    if args.log_path:
        logging.basicConfig(filename=args.log_path, level=args.log_level)
    else:
        logging.basicConfig(level=args.log_level)

    if not args.no_console_log:
        console = logging.StreamHandler()
        console.setLevel(args.log_level)
        logging.getLogger().addHandler(console)

    score_model(args.model_path, args.data_path, args.output_path)


if __name__ == "__main__":
    cli()
