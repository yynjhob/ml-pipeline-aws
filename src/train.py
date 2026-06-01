import os
import yaml
import json
import logging
import pickle

import mlflow
import mlflow.sklearn
import boto3

from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_data():
    logger.info("Loading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")
    train = dataset["train"].to_pandas()[["sentence", "label"]].dropna()
    val = dataset["validation"].to_pandas()[["sentence", "label"]].dropna()
    return train, val


def build_pipeline(config):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=config["model"]["max_features"],
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            C=config["model"]["C"],
            max_iter=config["model"]["max_iter"],
            random_state=42
        ))
    ])


def train(config_path="config.yaml"):
    config = load_config(config_path)

    bucket = config["aws"]["bucket"]
    region = config["aws"]["region"]

    # Set MLflow tracking to S3
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    train_df, val_df = load_data()

    with mlflow.start_run() as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Log parameters
        mlflow.log_params({
            "max_features": config["model"]["max_features"],
            "C":            config["model"]["C"],
            "max_iter":     config["model"]["max_iter"],
            "train_size":   len(train_df),
            "val_size":     len(val_df)
        })

        # Train
        logger.info("Training pipeline...")
        pipeline = build_pipeline(config)
        pipeline.fit(train_df["sentence"], train_df["label"])

        # Evaluate
        preds = pipeline.predict(val_df["sentence"])
        metrics = {
            "accuracy":  round(accuracy_score(val_df["label"], preds), 4),
            "f1":        round(f1_score(val_df["label"], preds), 4),
            "precision": round(precision_score(val_df["label"], preds), 4),
            "recall":    round(recall_score(val_df["label"], preds), 4)
        }
        logger.info(f"Metrics: {metrics}")
        mlflow.log_metrics(metrics)

        # Save metrics locally for CI gate
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to outputs/metrics.json")

        # Log model to MLflow
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=config["mlflow"]["model_name"]
        )

        # Save model locally and upload to S3
        model_path = "outputs/model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        s3 = boto3.client("s3", region_name=region)
        s3.upload_file(
            model_path,
            bucket,
            f"{config['aws']['model_prefix']}model.pkl"
        )
        logger.info(f"Model uploaded to s3://{bucket}/{config['aws']['model_prefix']}model.pkl")

        return metrics, run.info.run_id


if __name__ == "__main__":
    metrics, run_id = train()
    print(f"\nTraining complete.")
    print(f"Run ID: {run_id}")
    print(f"Metrics: {metrics}")