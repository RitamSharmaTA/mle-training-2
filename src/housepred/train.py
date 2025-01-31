import argparse
import logging
import os
import mlflow
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeRegressor


def train_model(input_path, output_path, model_type):
    """
    Trains a machine learning model using a dataset and saves the trained model.

    The function supports training three types of models: linear regression, decision tree,
    and random forest. It performs a stratified split of the data, processes the features,
    and then trains the selected model.

    Parameters
    ----------
    input_path : str
        The file path to the dataset used for training the model.
    output_path : str
        The file path where the trained model will be saved.
    model_type : str
        The type of model to train. Options are:
        - "linear" for Linear Regression,
        - "tree" for Decision Tree Regressor,
        - "forest" for Random Forest Regressor.

    Returns
    -------
    None
        This function does not return anything. It saves the trained model to the specified output path.

    Raises
    ------
    ValueError
        If an unsupported `model_type` is provided, a ValueError is raised.
    """
    logging.basicConfig(level=logging.INFO)
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
    for train_index, test_index in split.split(
        housing, housing["income_cat"]
    ):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Data preparation
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

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

    if model_type == "linear":
        logger.info("Training Linear Regression model...")
        model = LinearRegression()
        mlflow.log_param("model_type", "linear")
    elif model_type == "tree":
        logger.info("Training Decision Tree model...")
        model = DecisionTreeRegressor(random_state=42)
        mlflow.log_param("model_type", "tree")
        mlflow.log_param("random_state", 42)
    elif model_type == "forest":
        logger.info("Training Random Forest model...")
        model = RandomForestRegressor(random_state=42)
        mlflow.log_param("model_type", "forest")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_estimators", model.n_estimators)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train model and log training metrics
    model.fit(housing_prepared, housing_labels)
    train_predictions = model.predict(housing_prepared)
    train_mse = mean_squared_error(housing_labels, train_predictions)
    train_rmse = np.sqrt(train_mse)
    mlflow.log_metric("train_rmse", train_rmse)

    # Save model
    logger.info("Saving model...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    mlflow.log_artifact(output_path)
    logger.info(f"Model training completed. Model saved to {output_path}")


def cli():
    """
    Command-line interface for training a model.

    This function parses the command-line arguments, sets up logging, and invokes the
    `train_model` function to train the specified model using the provided dataset.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return anything. It orchestrates the command-line interface
        for model training.
    """
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
        help="Type of model to train: linear, tree, or forest",
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

    train_model(args.input_path, args.output_path, args.model_type)


if __name__ == "__main__":
    cli()
