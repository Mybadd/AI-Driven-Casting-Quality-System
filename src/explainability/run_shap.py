from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.data.load_data import load_csv
from src.features.schema import INPUT_FEATURES, CLASSIFICATION_TARGETS
from src.models.advanced.xgboost_classification import (
    create_xgboost_pipeline,
    encode_xgboost_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIRECTORY = PROJECT_ROOT / "reports" / "evaluation" / "shap"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def get_dataset_path() -> Path:
    csv_files = list(DATA_DIRECTORY.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV dataset found in {DATA_DIRECTORY}"
        )

    if len(csv_files) > 1:
        raise ValueError(
            f"Expected exactly one CSV dataset in {DATA_DIRECTORY}, "
            f"found {len(csv_files)}"
        )

    return csv_files[0]


def run_shap_for_target(
    dataframe: pd.DataFrame,
    target: str,
) -> None:

    print("\n" + "=" * 70)
    print(f"SHAP ANALYSIS: {target}")
    print("=" * 70)

    X = dataframe[INPUT_FEATURES]
    y = dataframe[target]

    y_encoded = encode_xgboost_target(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Create the same XGBoost pipeline used in the advanced experiment.
    pipeline = create_xgboost_pipeline(
        y_train=y_train,
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Transform test data using the fitted preprocessing stage.
    X_test_transformed = preprocessor.transform(X_test)

    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    X_test_transformed = pd.DataFrame(
        X_test_transformed,
        columns=feature_names,
        index=X_test.index,
    )

    print("\nTransformed features:")
    for feature in feature_names:
        print(f"  - {feature}")

    print("\nCreating TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(
        X_test_transformed
    )

    target_directory = OUTPUT_DIRECTORY / target.lower()
    target_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Global SHAP bar plot
    # ---------------------------------------------------------

    print("Creating global importance plot...")

    plt.figure()

    shap.plots.bar(
        shap_values,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    global_bar_path = (
        target_directory
        / "global_feature_importance.png"
    )

    plt.savefig(
        global_bar_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # SHAP beeswarm plot
    # ---------------------------------------------------------

    print("Creating beeswarm plot...")

    plt.figure()

    shap.plots.beeswarm(
        shap_values,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    beeswarm_path = (
        target_directory
        / "global_beeswarm.png"
    )

    plt.savefig(
        beeswarm_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # Save SHAP importance table
    # ---------------------------------------------------------

    mean_abs_shap = (
        pd.Series(
            abs(shap_values.values).mean(axis=0),
            index=feature_names,
            name="mean_abs_shap",
        )
        .sort_values(ascending=False)
    )

    importance_dataframe = (
        mean_abs_shap
        .reset_index()
        .rename(
            columns={
                "index": "feature",
            }
        )
    )

    importance_path = (
        target_directory
        / "global_feature_importance.csv"
    )

    importance_dataframe.to_csv(
        importance_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Representative local explanation
    # ---------------------------------------------------------

    print("Creating representative local explanation...")

    local_index = 0

    plt.figure()

    shap.plots.waterfall(
        shap_values[local_index],
        max_display=10,
        show=False,
    )

    plt.tight_layout()

    local_path = (
        target_directory
        / "local_waterfall_sample_0.png"
    )

    plt.savefig(
        local_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # Save test-case information
    # ---------------------------------------------------------

    sample_information = X_test.iloc[
        [local_index]
    ].copy()

    sample_information[
        "actual_target"
    ] = dataframe.loc[
        X_test.index[local_index],
        target,
    ]

    prediction = pipeline.predict(
        X_test.iloc[[local_index]]
    )[0]

    probability = pipeline.predict_proba(
        X_test.iloc[[local_index]]
    )[0][1]

    sample_information[
        "predicted_target"
    ] = "Yes" if prediction == 1 else "No"

    sample_information[
        "predicted_probability_yes"
    ] = probability

    sample_path = (
        target_directory
        / "local_sample_information.csv"
    )

    sample_information.to_csv(
        sample_path,
        index=False,
    )

    print("\nSaved SHAP outputs:")
    print(f"  {global_bar_path}")
    print(f"  {beeswarm_path}")
    print(f"  {importance_path}")
    print(f"  {local_path}")
    print(f"  {sample_path}")


def main() -> None:

    dataset_path = get_dataset_path()

    print(
        f"Loading dataset: {dataset_path.name}"
    )

    dataframe = load_csv(dataset_path)

    print(
        f"Dataset loaded successfully: "
        f"{len(dataframe)} rows"
    )

    for target in CLASSIFICATION_TARGETS:
        run_shap_for_target(
            dataframe=dataframe,
            target=target,
        )

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()