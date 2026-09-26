from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_data import load_csv
from src.features.schema import (
    CLASSIFICATION_TARGETS,
    INPUT_FEATURES,
)
from src.models.decision.oof_threshold_selection import (
    DecisionPolicy,
    run_oof_threshold_analysis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
    / "decision_policy"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

CANDIDATE_MODELS = (
    "balanced_logistic_regression",
    "xgboost",
)


# IMPORTANT:
# Do not put arbitrary engineering constraints here yet.
# We will decide these after examining the OOF results.
POLICIES = {
    "Defect": DecisionPolicy(),
    "Porosity": DecisionPolicy(),
    "Scrap": DecisionPolicy(),
}
POLICIES = {
    target: DecisionPolicy(
        min_recall=0.50,
        max_alert_rate=0.50,
    )
    for target in CLASSIFICATION_TARGETS
}

def find_single_dataset() -> Path:
    """Find the project CSV dataset."""

    csv_files = sorted(
        RAW_DATA_DIR.glob("*.csv")
    )

    if len(csv_files) != 1:

        raise RuntimeError(
            f"Expected exactly one CSV in "
            f"{RAW_DATA_DIR}, "
            f"found {len(csv_files)}"
        )

    return csv_files[0]


def main() -> None:

    dataset_path = find_single_dataset()

    print(
        f"Loading dataset: "
        f"{dataset_path}"
    )

    df = load_csv(
        dataset_path
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    for target in CLASSIFICATION_TARGETS:

        print("\n" + "=" * 80)
        print(f"TARGET: {target}")
        print("=" * 80)

        X = df[
            INPUT_FEATURES
        ].copy()

        y = df[
            target
        ].copy()

        # ---------------------------------------------------------
        # IMPORTANT:
        # Create the final held-out test set first.
        #
        # The test set is NOT used during threshold selection.
        # ---------------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        print(
            f"Training rows: {len(X_train)}"
        )

        print(
            f"Test rows: {len(X_test)}"
        )

        # Save a manifest documenting the split.
        split_manifest = pd.DataFrame(
            [
                {
                    "target": target,
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE,
                    "threshold_selection_uses_test_set": False,
                }
            ]
        )

        split_manifest.to_csv(
            OUTPUT_DIR
            / (
                f"{target.lower()}"
                "__split_manifest.csv"
            ),
            index=False,
        )

        for model_name in CANDIDATE_MODELS:

            print("\n" + "-" * 80)
            print(
                f"MODEL: {model_name}"
            )
            print("-" * 80)

            threshold_results, selected_threshold = (
                run_oof_threshold_analysis(
                    X_train=X_train,
                    y_train=y_train,
                    target_name=target,
                    model_name=model_name,
                    output_dir=OUTPUT_DIR,
                    policy=POLICIES[target],
                )
            )

            # -----------------------------------------------------
            # This is ONLY an exploratory statistic.
            #
            # We do NOT use it to select the final threshold.
            # -----------------------------------------------------

            best_f1 = threshold_results.loc[
                threshold_results["f1"].idxmax()
            ]

            summaries.append(
                {
                    "target": target,
                    "model": model_name,

                    "best_f1_threshold_exploratory": (
                        best_f1["threshold"]
                    ),

                    "best_f1_exploratory": (
                        best_f1["f1"]
                    ),

                    "best_f1_precision": (
                        best_f1["precision"]
                    ),

                    "best_f1_recall": (
                        best_f1["recall"]
                    ),

                    "best_f1_alert_rate": (
                        best_f1["alert_rate"]
                    ),

                    "policy_threshold_selected": (
                        None
                        if selected_threshold is None
                        else selected_threshold["threshold"]
                    ),

                    "note": (
                        "Best-F1 threshold is exploratory only. "
                        "Final threshold must follow the approved "
                        "engineering decision policy."
                    ),
                }
            )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = (
        OUTPUT_DIR
        / "oof_threshold_model_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("OOF THRESHOLD ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"Results saved to:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        "\nNo final threshold has been locked."
    )

    print(
        "Threshold selection will be based on "
        "the engineering decision policy."
    )


if __name__ == "__main__":
    main()