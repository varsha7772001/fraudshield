import pandas as pd


class TargetAnalyzer:
    """
    Analyzes the target variable of the dataset.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe

    def analyze(self) -> None:
        target_counts = self.dataframe["isFraud"].value_counts()
        target_percentages = (
            self.dataframe["isFraud"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        print("\n" + "=" * 60)
        print("Target Variable Analysis")
        print("=" * 60)

        print(f"Legitimate Transactions : {target_counts[0]:,}")
        print(f"Fraud Transactions      : {target_counts[1]:,}")
        print()
        print(f"Legitimate Percentage   : {target_percentages[0]}%")
        print(f"Fraud Percentage        : {target_percentages[1]}%")