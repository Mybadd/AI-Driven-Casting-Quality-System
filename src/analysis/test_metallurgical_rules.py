from src.analysis.metallurgical_rules import build_quality_review


def main() -> None:
    influential_features = [
        ("Mold_Moisture", 0.209),
        ("Riser", 0.190),
        ("Pour_Temp", 0.163),
    ]

    review = build_quality_review(
        target="Defect",
        influential_features=influential_features,
    )

    print("\nTarget:", review["target"])
    print("\nStatement:")
    print(review["statement"])

    print("\nFactors:")

    for factor in review["factors"]:
        print("\n-----------------------------")
        print("Factor:", factor["factor"])
        print("Importance:", factor["importance"])
        print("Interpretation:", factor["interpretation"])
        print("Review:", factor["review"])


if __name__ == "__main__":
    main()