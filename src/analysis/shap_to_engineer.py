"""
Connect SHAP global feature importance to the metallurgical rule engine.
"""

from pathlib import Path

import pandas as pd

from src.analysis.metallurgical_rules import build_quality_review


FEATURE_NAME_MAP = {
    "numerical__Pour_Temp": "Pour_Temp",
    "numerical__Mold_Moisture": "Mold_Moisture",
    "numerical__Cooling_Time": "Cooling_Time",
    "numerical__Riser": "Riser",
    "categorical__Alloy_Al-Mg": "Alloy",
    "categorical__Alloy_Al-Si": "Alloy",
    "categorical__Alloy_Al-Si-Cu": "Alloy",
    "categorical__Alloy_Al-Si-Mg": "Alloy",
}


def load_shap_importance(csv_path: Path) -> pd.DataFrame:
    """Load and validate a SHAP global importance CSV."""

    dataframe = pd.read_csv(csv_path)

    required_columns = {"feature", "mean_abs_shap"}

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return dataframe


def convert_shap_features(dataframe: pd.DataFrame) -> list[tuple[str, float]]:
    """
    Convert transformed SHAP feature names into original engineering features.

    Features are returned in descending SHAP importance order.
    """

    converted = []

    for _, row in dataframe.iterrows():
        transformed_name = row["feature"]

        if transformed_name not in FEATURE_NAME_MAP:
            continue

        engineering_name = FEATURE_NAME_MAP[transformed_name]

        converted.append(
            (
                engineering_name,
                float(row["mean_abs_shap"]),
            )
        )

    return converted


def generate_engineering_review(
    target: str,
    shap_csv_path: Path,
    top_n: int = 4,
) -> dict:
    """
    Generate an engineering review from a SHAP importance CSV.

    Duplicate engineering features are combined by retaining their
    individual SHAP contributions and then aggregating them.
    """

    shap_data = load_shap_importance(shap_csv_path)

    converted_features = convert_shap_features(shap_data)

    aggregated: dict[str, float] = {}

    for feature_name, importance in converted_features:
        aggregated[feature_name] = (
            aggregated.get(feature_name, 0.0) + importance
        )

    ranked_features = sorted(
        aggregated.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_features = ranked_features[:top_n]

    return build_quality_review(
        target=target,
        influential_features=top_features,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    shap_root = project_root / "reports" / "evaluation" / "shap"
    output_dir = project_root / "reports" / "evaluation" / "engineering"
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "Defect": shap_root / "defect" / "global_feature_importance.csv",
        "Porosity": shap_root / "porosity" / "global_feature_importance.csv",
        "Scrap": shap_root / "scrap" / "global_feature_importance.csv",
    }
    all_reviews = {}
    for target, csv_path in targets.items():
        review = generate_engineering_review(
            target=target,
            shap_csv_path=csv_path,
        )
        all_reviews[target] = review
        print("\n" + "=" * 60)
        print(f"TARGET: {review['target']}")
        print("=" * 60)

        print("\nEngineering interpretation:")
        print(review["statement"])

        for factor in review["factors"]:
            print("\n-----------------------------")
            print(f"Factor: {factor['factor']}")
            print(f"SHAP importance: {factor['importance']:.6f}")
            print(f"Interpretation: {factor['interpretation']}")
            print(f"Review: {factor['review']}")
    import json

    output_path = output_dir / "engineering_review.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            all_reviews,
            file,
            indent=2,
        )

    print(f"\nSaved engineering review: {output_path}")

if __name__ == "__main__":
    main()