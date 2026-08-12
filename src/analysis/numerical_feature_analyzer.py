from dataclasses import dataclass

import pandas as pd


@dataclass
class NumericalFeatureSummary:
    feature_name: str
    data_type: str

    total_count: int

    missing_count: int
    missing_percentage: float

    minimum: float
    maximum: float

    mean: float
    median: float

    standard_deviation: float

    skewness: float
    kurtosis: float

    unique_values: int


class NumericalFeatureAnalyzer:

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe

    def analyze(
        self,
        feature_name: str,
    ) -> NumericalFeatureSummary:

        self._validate_feature(feature_name)

        series = self.dataframe[feature_name]

        total_count = len(series)

        missing_count = int(series.isnull().sum())

        missing_percentage = round(
            float(series.isnull().mean() * 100),
            2,
        )

        minimum = float(series.min())

        maximum = float(series.max())

        mean = round(float(series.mean()), 2)

        median = round(float(series.median()), 2)

        standard_deviation = round(
            float(series.std()),
            2,
        )

        skewness = round(
            float(series.skew()),
            2,
        )

        kurtosis = round(
            float(series.kurt()),
            2,
        )

        unique_values = int(series.nunique())

        return NumericalFeatureSummary(
            feature_name=feature_name,
            data_type=str(series.dtype),
            total_count=total_count,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            median=median,
            standard_deviation=standard_deviation,
            skewness=skewness,
            kurtosis=kurtosis,
            unique_values=unique_values,
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