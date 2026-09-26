"""Engineering review service combining SHAP and metallurgical interpretation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.analysis.shap_to_engineer import generate_engineering_review


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAP_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
    / "shap"
)


TARGET_TO_SHAP_FILE = {
    "Defect": SHAP_DIRECTORY / "defect" / "global_feature_importance.csv",
    "Porosity": SHAP_DIRECTORY / "porosity" / "global_feature_importance.csv",
    "Scrap": SHAP_DIRECTORY / "scrap" / "global_feature_importance.csv",
}


def get_engineering_review(
    target: str,
    top_n: int = 4,
) -> Dict:
    """Return SHAP-informed engineering factors for a target."""

    if target not in TARGET_TO_SHAP_FILE:
        raise ValueError(
            f"Unsupported target: {target}. "
            f"Supported targets: {list(TARGET_TO_SHAP_FILE)}"
        )

    shap_path = TARGET_TO_SHAP_FILE[target]

    if not shap_path.exists():
        raise FileNotFoundError(
            f"SHAP importance file not found: {shap_path}"
        )

    return generate_engineering_review(
        target=target,
        shap_csv_path=shap_path,
        top_n=top_n,
    )


def get_all_engineering_reviews(top_n: int = 4) -> Dict[str, Dict]:
    """Return engineering reviews for all supported quality targets."""

    return {
        target: get_engineering_review(target, top_n=top_n)
        for target in TARGET_TO_SHAP_FILE
    }