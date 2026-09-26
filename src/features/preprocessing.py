from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.features.schema import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from sklearn.compose import ColumnTransformer

def create_tree_preprocessor() -> ColumnTransformer:
    """
    Create preprocessing for tree-based models.

    Numerical features are passed without scaling because
    tree-based models do not require feature scaling.

    Categorical features are one-hot encoded.
    """

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
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


def create_scaled_preprocessor() -> ColumnTransformer:
    """
    Create preprocessing for scale-sensitive models.

    Numerical features are standardized and categorical
    features are one-hot encoded.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                numeric_pipeline,
                NUMERICAL_FEATURES,
            ),
        ]
    )
def create_scaled_preprocessor_for_features(
    numerical_features: list[str],
) -> ColumnTransformer:
    """
    Create a scaled preprocessor for a custom numerical feature set.

    Alloy remains one-hot encoded while the supplied numerical features
    are standardized.
    """

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["Alloy"],
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_features,
            ),
        ]
    )