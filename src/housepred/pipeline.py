import argparse
import logging
import os

import mlflow

from housepred.ingest_data import main as ingest_data  # Import the ingestion function
from housepred.score import score_model
from housepred.train import train_model


def run_pipeline(input_path, model_path, output_path, model_type):
    """
    Orchestrates the complete ML pipeline with MLflow tracking,
    including data ingestion.
    """
    with mlflow.start_run(run_name="housepred_pipeline") as parent_run:
        mlflow.log_param("input_path", input_path)
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("output_path", output_path)
        mlflow.log_param("model_type", model_type)

        print("Starting pipeline with:")
        print(f"Input path: {input_path}")
        print(f"Model path: {model_path}")
        print(f"Output path: {output_path}")
        print(f"Model type: {model_type}")

        # ✅ Ingest Data Step
        with mlflow.start_run(run_name="data_ingestion", nested=True):
            if not os.path.exists(input_path):  # Only fetch if file is missing
                print("Ingesting data...")
                ingest_data(os.path.dirname(input_path))
            else:
                print("Data already exists, skipping ingestion.")

        # ✅ Train Model Step
        with mlflow.start_run(run_name="model_training", nested=True):
            train_model(input_path, model_path, model_type)

        # ✅ Score Model Step
        with mlflow.start_run(run_name="model_scoring", nested=True):
            score_model(model_path, input_path, output_path)


def cli():
    parser = argparse.ArgumentParser(
        description="Run the complete ML pipeline with MLflow tracking"
    )
    parser.add_argument(
        "--input-path", type=str, required=True, help="Input dataset path"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Model output path"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Results output path"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["linear", "tree", "forest"],
        required=True,
        help="Type of model to train",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    run_pipeline(args.input_path, args.model_path, args.output_path, args.model_type)


if __name__ == "__main__":
    cli()
