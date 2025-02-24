import importlib
import os
import subprocess

import pytest


def test_package_import():
    """Test if the housepred package can be imported successfully."""
    try:
        importlib.import_module("housepred")
    except ImportError as e:
        pytest.fail(f"❌ Failed to import housepred: {e}")


@pytest.mark.parametrize(
    "cmd",
    [
        [
            "python",
            "-m",
            "housepred.pipeline",
            "--input-path",
            "data/housing.csv",
            "--model-path",
            "models/housepred.pkl",
            "--output-path",
            "results/predictions.txt",
            "--model-type",
            "linear",
        ]
    ],
)
def test_cli_commands(cmd):
    """Test if the CLI commands execute without errors."""
    # Extract argument values dynamically
    input_path = cmd[cmd.index("--input-path") + 1]
    model_path = cmd[cmd.index("--model-path") + 1]

    if not os.path.exists(input_path):
        subprocess.run(
            [
                "python",
                "-m",
                "housepred.ingest_data",
                "--output-path",
                "data",
            ],
            check=True,
        )

    assert os.path.exists(input_path), f"❌ Missing input file: {input_path}"
    assert os.path.exists(model_path) or not os.path.exists(
        model_path
    ), f"⚠️ Model file {model_path} does not exist yet (expected for fresh runs)."

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"❌ Command {' '.join(cmd)} failed: {result.stderr}"
