import argparse
import logging
import os
import tarfile

import pandas as pd
from six.moves import urllib

from housepred.logger import setup_logging


def fetch_housing_data(housing_url, housing_path):
    """
    Fetches housing data from a specified URL and extracts it to a local path.

    Parameters
    ----------
    housing_url : str
        The URL where the housing data archive is located.
    housing_path : str
        The local directory where the data should be saved and extracted.

    Returns
    -------
    None
        This function does not return anything, it just saves and extracts the data.
    """
    logger = logging.getLogger(__name__)
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    logger.info(f"Downloading data from {housing_url} to {tgz_path}...")
    urllib.request.urlretrieve(housing_url, tgz_path)
    logger.info("Extracting data...")
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    logger.info("Data extraction complete.")


def load_housing_data(housing_path):
    """
    Loads the housing data CSV file from a specified path into a pandas DataFrame.

    Parameters
    ----------
    housing_path : str
        The directory where the housing CSV file is located.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the housing data from the CSV file.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {csv_path}...")
    return pd.read_csv(csv_path)


def main(output_path="data"):
    """
    Main function to fetch and load the housing data.

    Parameters
    ----------
    output_path : str, optional
        The directory where the data should be saved and extracted.
        The default is "data".

    Returns
    -------
    None
        This function does not return anything, it just fetches and loads the data.
    """
    logger = logging.getLogger(__name__)
    logger.info("Fetching housing data...")
    HOUSING_URL = (
        "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
        "datasets/housing/housing.tgz"
    )
    fetch_housing_data(HOUSING_URL, output_path)
    logger.info("Data fetched successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data for housing dataset.")
    parser.add_argument(
        "--output-path",
        type=str,
        default="data",
        help="Output folder/file path",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    parser.add_argument("--log-path", type=str, help="Log file path")
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable console logging",
    )
    args = parser.parse_args()

    # Debug statements to verify logging setup
    print(f"Log Level: {args.log_level}")
    print(f"Log Path: {args.log_path}")
    print(f"No Console Log: {args.no_console_log}")

    setup_logging(
        log_level=args.log_level,
        log_path=args.log_path,
        no_console_log=args.no_console_log,
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting model ingest...")

    main(args.output_path)
