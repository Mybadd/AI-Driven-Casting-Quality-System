"""
Reusable baseline regression training utilities.

Supports the Yield prediction target.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.preprocessing import (
    create_scaled_preprocessor,
    create_tree_preprocessor,
)
from src.models.core.model_factory import (
    create_regression_model,
)


SCALED_REGRESSION_MODELS = {
    "linear_regression",
}


TREE_REGRESSION_MODELS = {
    "random_forest_regressor",
}


def create_regression_pipeline(
    model_name: str,
) -> Pipeline:
    """
    Create a complete preprocessing and regression pipeline.

    The preprocessing strategy is automatically selected according
    to the model requirements.
    """

    model = create_regression_model(
        model_name=model_name,
    )

    if model_name in SCALED_REGRESSION_MODELS:
        preprocessor = create_scaled_preprocessor()

    elif model_name in TREE_REGRESSION_MODELS:
        preprocessor = create_tree_preprocessor()

    else:
        raise ValueError(
            f"Unsupported regression pipeline: {model_name}"
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


def train_regression_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Train a baseline regression model.

    The preprocessor is fitted only on the training data because
    it is part of the sklearn pipeline.
    """

    pipeline = create_regression_pipeline(
        model_name=model_name,
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline


def predict_regression(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
) -> pd.Series:
    """
    Generate regression predictions using a trained pipeline.
    """

    predictions = pipeline.predict(
        X_test,
    )

    return pd.Series(
        predictions,
        index=X_test.index,
        name="prediction",
    )   