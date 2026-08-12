from dataclasses import dataclass

import pandas as pd


@dataclass
class MissingValueTargetSummary:
    feature_name: str

    missing_count: int
    missing_percentage: float

    available_count: int
    available_percentage: float

    missing_fraud_count: int
    available_fraud_count: int

    missing_legitimate_count: int
    available_legitimate_count: int

    missing_fraud_rate: float
    available_fraud_rate: float

    fraud_rate_difference: float


class MissingValueTargetAnalyzer:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str = "isFraud",
    ) -> None:

        self.dataframe = dataframe
        self.target_column = target_column

    def analyze(
        self,
        feature_name: str,
    ) -> MissingValueTargetSummary:

        self._validate_feature(feature_name)

        missing = self.dataframe[
            self.dataframe[feature_name].isna()
        ]

        available = self.dataframe[
            self.dataframe[feature_name].notna()
        ]

        total_rows = len(self.dataframe)

        missing_count = len(missing)
        available_count = len(available)

        missing_percentage = round(
            (missing_count / total_rows) * 100,
            2,
        )

        available_percentage = round(
            (available_count / total_rows) * 100,
            2,
        )

        missing_fraud_count = int(
            missing[self.target_column].sum()
        )

        available_fraud_count = int(
            available[self.target_column].sum()
        )

        missing_legitimate_count = (
            missing_count - missing_fraud_count
        )

        available_legitimate_count = (
            available_count - available_fraud_count
        )

        missing_fraud_rate = (
            round(
                (missing_fraud_count / missing_count) * 100,
                2,
            )
            if missing_count > 0
            else 0.0
        )

        available_fraud_rate = (
            round(
                (available_fraud_count / available_count) * 100,
                2,
            )
            if available_count > 0
            else 0.0
        )

        fraud_rate_difference = round(
            abs(
                available_fraud_rate -
                missing_fraud_rate
            ),
            2,
        )

        return MissingValueTargetSummary(
            feature_name=feature_name,

            missing_count=missing_count,
            missing_percentage=missing_percentage,

            available_count=available_count,
            available_percentage=available_percentage,

            missing_fraud_count=missing_fraud_count,
            available_fraud_count=available_fraud_count,

            missing_legitimate_count=missing_legitimate_count,
            available_legitimate_count=available_legitimate_count,

            missing_fraud_rate=missing_fraud_rate,
            available_fraud_rate=available_fraud_rate,

            fraud_rate_difference=fraud_rate_difference,
        )

    def analyze_all(
        self,
        minimum_missing_count: int = 1000,
    ) -> list[MissingValueTargetSummary]:

        summaries = []

        for feature_name in self.dataframe.columns:

            if feature_name == self.target_column:
                continue

            summary = self.analyze(feature_name)

            if summary.missing_count < minimum_missing_count:
                continue

            summaries.append(summary)

        summaries.sort(
            key=lambda x: (
                x.fraud_rate_difference,
                x.missing_count,
            ),
            reverse=True,
        )

        return summaries

    def _validate_feature(
        self,
        feature_name: str,
    ) -> None:

        if feature_name not in self.dataframe.columns:
            raise ValueError(
                f"Feature '{feature_name}' does not exist."
            )

        if self.target_column not in self.dataframe.columns:
            raise ValueError(
                f"Target column '{self.target_column}' does not exist."
            )