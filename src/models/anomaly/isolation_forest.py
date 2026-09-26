"""
Isolation Forest anomaly detection for casting process conditions.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from pathlib import Path

import joblib

RANDOM_STATE = 42

ANOMALY_CONTAMINATION = 0.05


def create_anomaly_pipeline() -> Pipeline:
    """
    Create an Isolation Forest pipeline.

    The model uses only the approved process input features.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                "passthrough",
                NUMERICAL_FEATURES,
            ),
        ]
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=ANOMALY_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_anomaly_model(
    X_train: pd.DataFrame,
) -> Pipeline:
    """Fit the anomaly detector using historical process inputs."""

    pipeline = create_anomaly_pipeline()
    pipeline.fit(X_train)

    return pipeline

def save_anomaly_model(
    pipeline: Pipeline,
    model_path: Path,
) -> None:
    """Save the trained anomaly detection pipeline."""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    
def predict_anomalies(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return anomaly labels and scores.

    Isolation Forest returns:
        1  -> normal
       -1  -> anomaly
    """

    labels = pipeline.predict(X)
    scores = pipeline.decision_function(X)

    return pd.DataFrame(
        {
            "anomaly_label": labels,
            "anomaly_status": [
                "Anomaly" if label == -1 else "Normal"
                for label in labels
            ],
            "anomaly_score": scores,
        },
        index=X.index,
    )