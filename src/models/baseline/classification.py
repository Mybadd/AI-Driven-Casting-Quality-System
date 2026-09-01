"""
Reusable baseline classification training utilities.

Supports the Defect, Porosity, and Scrap prediction targets.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.preprocessing import (
    create_scaled_preprocessor,
    create_tree_preprocessor,
)
from src.models.core.model_factory import (
    create_classification_model,
)


SCALED_CLASSIFICATION_MODELS = {
    "logistic_regression",
}


TREE_CLASSIFICATION_MODELS = {
    "random_forest_classifier",
}


def create_classification_pipeline(
    model_name: str,
) -> Pipeline:
    """
    Create a complete preprocessing and classification pipeline.

    The preprocessing strategy is automatically selected according
    to the model requirements.

    Parameters
    ----------
    model_name:
        Name of the classification model.

    Returns
    -------
    Pipeline
        Complete preprocessing and model pipeline.

    Raises
    ------
    ValueError
        If the model is not supported.
    """

    model = create_classification_model(
        model_name=model_name,
    )

    if model_name in SCALED_CLASSIFICATION_MODELS:
        preprocessor = create_scaled_preprocessor()

    elif model_name in TREE_CLASSIFICATION_MODELS:
        preprocessor = create_tree_preprocessor()

    else:
        raise ValueError(
            f"Unsupported classification pipeline: "
            f"{model_name}"
        )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


def train_classification_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Train a baseline classification model.

    The preprocessor is fitted only on the training data because
    it is part of the sklearn pipeline.

    Parameters
    ----------
    model_name:
        Name of the classification model.
    X_train:
        Training feature dataframe.
    y_train:
        Training target values.

    Returns
    -------
    Pipeline
        Fitted classification pipeline.
    """

    pipeline = create_classification_pipeline(
        model_name=model_name,
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline


def predict_classification(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
) -> pd.Series:
    """
    Generate predictions using a trained classification pipeline.

    Parameters
    ----------
    pipeline:
        Fitted classification pipeline.
    X_test:
        Testing feature dataframe.

    Returns
    -------
    pd.Series
        Predicted target values.
    """

    predictions = pipeline.predict(
        X_test,
    )

    return pd.Series(
        predictions,
        index=X_test.index,
        name="prediction",
    )


def predict_classification_probabilities(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
) -> Any:
    """
    Generate class probability predictions.

    Parameters
    ----------
    pipeline:
        Fitted classification pipeline.
    X_test:
        Testing feature dataframe.

    Returns
    -------
    Any
        Probability predictions produced by the model.
    """

    return pipeline.predict_proba(
        X_test,
    )