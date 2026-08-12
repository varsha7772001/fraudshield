from dataclasses import dataclass

import pandas as pd


@dataclass
class NumericalTargetSummary:
    feature_name: str

    fraud_count: int
    legitimate_count: int

    fraud_mean: float
    legitimate_mean: float

    fraud_median: float
    legitimate_median: float

    fraud_minimum: float
    legitimate_minimum: float

    fraud_maximum: float
    legitimate_maximum: float

    fraud_standard_deviation: float
    legitimate_standard_deviation: float


class NumericalTargetAnalyzer:

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
    ) -> NumericalTargetSummary:

        self._validate_feature(feature_name)

        fraud = self.dataframe[
            self.dataframe[self.target_column] == 1
        ][feature_name]

        legitimate = self.dataframe[
            self.dataframe[self.target_column] == 0
        ][feature_name]

        return NumericalTargetSummary(
            feature_name=feature_name,

            fraud_count=len(fraud),
            legitimate_count=len(legitimate),

            fraud_mean=round(float(fraud.mean()), 2),
            legitimate_mean=round(float(legitimate.mean()), 2),

            fraud_median=round(float(fraud.median()), 2),
            legitimate_median=round(float(legitimate.median()), 2),

            fraud_minimum=round(float(fraud.min()), 2),
            legitimate_minimum=round(float(legitimate.min()), 2),

            fraud_maximum=round(float(fraud.max()), 2),
            legitimate_maximum=round(float(legitimate.max()), 2),

            fraud_standard_deviation=round(
                float(fraud.std()),
                2,
            ),
            legitimate_standard_deviation=round(
                float(legitimate.std()),
                2,
            ),
        )

    def _validate_feature(
        self,
        feature_name: str,
    ) -> None:

        if feature_name not in self.dataframe.columns:
            raise ValueError(
                f"Feature '{feature_name}' does not exist."
            )

        if not pd.api.types.is_numeric_dtype(
            self.dataframe[feature_name]
        ):
            raise TypeError(
                f"'{feature_name}' is not a numerical feature."
            )

        if self.target_column not in self.dataframe.columns:
            raise ValueError(
                f"Target column '{self.target_column}' does not exist."
            )