from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap


def get_transformed_feature_names(preprocessor) -> list[str]:
    """
    Return feature names after preprocessing.
    """
    return list(preprocessor.get_feature_names_out())


def prepare_xgboost_explanation_data(
    pipeline,
    X_data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Transform raw input data using the already-fitted preprocessing
    stage of an ML pipeline.

    Returns:
        transformed_data: DataFrame suitable for SHAP
        feature_names: transformed feature names
    """
    preprocessor = pipeline.named_steps["preprocessor"]

    transformed = preprocessor.transform(X_data)
    feature_names = get_transformed_feature_names(preprocessor)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed_data = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=X_data.index,
    )

    return transformed_data, feature_names


def create_tree_explainer(pipeline):
    """
    Create a TreeExplainer for a fitted XGBoost pipeline.
    """
    model = pipeline.named_steps["model"]
    return shap.TreeExplainer(model)


def calculate_shap_values(
    pipeline,
    X_data: pd.DataFrame,
) -> tuple[shap.Explanation, pd.DataFrame]:
    """
    Calculate SHAP values for an already-fitted tree-based pipeline.
    """
    transformed_data, _ = prepare_xgboost_explanation_data(
        pipeline,
        X_data,
    )

    explainer = create_tree_explainer(pipeline)

    shap_values = explainer(transformed_data)

    return shap_values, transformed_data


def save_global_importance_plot(
    shap_values: shap.Explanation,
    output_path: Path,
    max_display: int = 15,
) -> None:
    """
    Save a global SHAP feature-importance bar plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.plots.bar(
        shap_values,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_beeswarm_plot(
    shap_values: shap.Explanation,
    output_path: Path,
    max_display: int = 15,
) -> None:
    """
    Save a SHAP beeswarm plot showing global feature influence.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.plots.beeswarm(
        shap_values,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_local_waterfall_plot(
    shap_values: shap.Explanation,
    sample_index: int,
    output_path: Path,
    max_display: int = 10,
) -> None:
    """
    Save a waterfall plot for one prediction.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.plots.waterfall(
        shap_values[sample_index],
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()