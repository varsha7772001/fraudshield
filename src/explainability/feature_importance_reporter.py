from src.explainability.models import FeatureImportance


class FeatureImportanceReporter:

    def report(
        self,
        importances: list[FeatureImportance],
        top_n: int = 20,
    ) -> None:

        print("\n" + "=" * 80)
        print("Top Feature Importances")
        print("=" * 80)

        print(
            f"{'Rank':<8}"
            f"{'Feature':<25}"
            f"{'Importance':>15}"
        )

        print("-" * 80)

        for importance in importances[:top_n]:

            print(
                f"{importance.rank:<8}"
                f"{importance.feature_name:<25}"
                f"{importance.importance:>15.6f}"
            )