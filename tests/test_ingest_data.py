import os
from unittest import mock

import pandas as pd
import pytest

from housepred.ingest_data import fetch_housing_data, load_housing_data


@pytest.fixture
def temp_housing_dir(tmp_path):
    """Fixture to create a temporary directory for housing data."""
    temp_dir = tmp_path / "housing"
    temp_dir.mkdir()
    return temp_dir


@mock.patch("housepred.ingest_data.urllib.request.urlretrieve")
@mock.patch("housepred.ingest_data.tarfile.open")
def test_fetch_housing_data(mock_tar_open, mock_urlretrieve, temp_housing_dir):
    """
    Test that fetch_housing_data creates the required files and directories
    without actually downloading files.
    """
    housing_url = "https://example.com/housing.tgz"
    fetch_housing_data(housing_url, str(temp_housing_dir))

    assert os.path.exists(temp_housing_dir)

    mock_urlretrieve.assert_called_once_with(
        housing_url, os.path.join(temp_housing_dir, "housing.tgz")
    )

    mock_tar_open.assert_called_once_with(os.path.join(temp_housing_dir, "housing.tgz"))


def test_load_housing_data(temp_housing_dir):
    """
    Test that load_housing_data correctly loads a CSV file into a DataFrame.
    """

    csv_path = os.path.join(temp_housing_dir, "housing.csv")
    sample_data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    sample_data.to_csv(csv_path, index=False)

    df = load_housing_data(str(temp_housing_dir))

    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]
    assert df.iloc[0]["col1"] == 1
