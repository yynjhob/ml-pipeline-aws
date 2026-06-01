import os
import json
import yaml
import logging
import pickle
import numpy as np

import boto3
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix
)
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(config):
    logger.info("Loading model from outputs/model.pkl...")
    with open("outputs/model.pkl", "rb") as f:
        return pickle.load(f)


def load_validation_data():
    logger.info("Loading SST-2 validation data...")
    dataset = load_dataset("glue", "sst2")
    val = dataset["validation"].to_pandas()[["sentence", "label"]].dropna()
    return val


def compute_metrics(model, val_df):
    preds = model.predict(val_df["sentence"])
    metrics = {
        "accuracy":  round(accuracy_score(val_df["label"], preds), 4),
        "f1":        round(f1_score(val_df["label"], preds), 4),
        "precision": round(precision_score(val_df["label"], preds), 4),
        "recall":    round(recall_score(val_df["label"], preds), 4)
    }
    cm = confusion_matrix(val_df["label"], preds)
    logger.info(f"Confusion matrix:\n{cm}")
    return metrics


def check_drift(val_df, config):
    """
    Simple input drift check — compare sentence length distribution
    of validation set against a stored baseline using KS test.
    """
    logger.info("Running drift check...")
    val_lengths = val_df["sentence"].str.len().values

    baseline_path = "outputs/baseline_lengths.npy"

    if not os.path.exists(baseline_path):
        # First run — save as baseline
        np.save(baseline_path, val_lengths)
        logger.info("No baseline found — saving current distribution as baseline.")
        return {"drift_detected": False, "ks_statistic": 0.0, "p_value": 1.0}

    baseline_lengths = np.load(baseline_path)
    ks_stat, p_value = stats.ks_2samp(baseline_lengths, val_lengths)

    drift_detected = p_value < 0.05
    result = {
        "drift_detected": drift_detected,
        "ks_statistic":   round(float(ks_stat), 4),
        "p_value":        round(float(p_value), 4)
    }
    logger.info(f"Drift check result: {result}")
    return result


def promotion_gate(metrics, config):
    """
    Only promote model to production if accuracy exceeds threshold.
    """
    threshold = config["evaluation"]["accuracy_threshold"]
    passed = metrics["accuracy"] >= threshold
    logger.info(f"Promotion gate: accuracy {metrics['accuracy']} >= {threshold} → {'PASSED' if passed else 'FAILED'}")
    return passed


def save_results(metrics, drift, passed):
    os.makedirs("outputs", exist_ok=True)

    # Convert all drift values to native Python types
    drift_clean = {
        "drift_detected": bool(drift["drift_detected"]),
        "ks_statistic":   float(drift["ks_statistic"]),
        "p_value":        float(drift["p_value"])
    }

    results = {
        "metrics":        metrics,
        "drift":          drift_clean,
        "promotion_gate": "PASSED" if passed else "FAILED"
    }

    with open("outputs/evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Evaluation results saved to outputs/evaluation.json")
    return results

def evaluate(config_path="config.yaml"):
    config = load_config(config_path)

    model  = load_model(config)
    val_df = load_validation_data()

    metrics = compute_metrics(model, val_df)
    drift   = check_drift(val_df, config)
    passed  = promotion_gate(metrics, config)

    results = save_results(metrics, drift, passed)

    logger.info(f"\nEvaluation complete.")
    logger.info(f"Metrics:        {metrics}")
    logger.info(f"Drift:          {drift}")
    logger.info(f"Promotion gate: {'PASSED ✓' if passed else 'FAILED ✗'}")

    if not passed:
        raise SystemExit(
            f"Model failed promotion gate. "
            f"Accuracy {metrics['accuracy']} below threshold "
            f"{config['evaluation']['accuracy_threshold']}"
        )

    return results


if __name__ == "__main__":
    evaluate()