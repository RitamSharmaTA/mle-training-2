import argparse
import logging
import os
import tarfile

import pandas as pd
from six.moves import urllib


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
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


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
    return pd.read_csv(csv_path)


def main(output_path="data"):  # Add default value
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
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)  # Fix the name reference
    logger.info("Fetching housing data...")

    HOUSING_URL = (
        "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
        "datasets/housing/housing.tgz"  # Add this URL
    )

    fetch_housing_data(HOUSING_URL, output_path)  # Fix the function call
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

    main(args.output_path)
