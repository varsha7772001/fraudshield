from dataclasses import dataclass

import pandas as pd


@dataclass
class CategoricalFeatureSummary:
    feature_name: str
    data_type: str

    total_count: int

    missing_count: int
    missing_percentage: float

    unique_values: int

    most_frequent_value: str
    most_frequent_count: int
    most_frequent_percentage: float

    cardinality: str


class CategoricalFeatureAnalyzer:

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe

    def analyze(
        self,
        feature_name: str,
    ) -> CategoricalFeatureSummary:

        self._validate_feature(feature_name)

        series = self.dataframe[feature_name]

        total_count = len(series)

        missing_count = int(series.isnull().sum())

        missing_percentage = round(
            float(series.isnull().mean() * 100),
            2,
        )

        unique_values = int(series.nunique(dropna=True))

        value_counts = series.value_counts(dropna=True)

        most_frequent_value = str(value_counts.index[0])

        most_frequent_count = int(value_counts.iloc[0])

        most_frequent_percentage = round(
            (most_frequent_count / total_count) * 100,
            2,
        )

        cardinality = self._get_cardinality(unique_values)

        return CategoricalFeatureSummary(
            feature_name=feature_name,
            data_type=str(series.dtype),
            total_count=total_count,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            unique_values=unique_values,
            most_frequent_value=most_frequent_value,
            most_frequent_count=most_frequent_count,
            most_frequent_percentage=most_frequent_percentage,
            cardinality=cardinality,
        )

    def _validate_feature(
        self,
        feature_name: str,
    ) -> None:

        if feature_name not in self.dataframe.columns:
            raise ValueError(
                f"Feature '{feature_name}' does not exist."
            )

        if pd.api.types.is_numeric_dtype(
            self.dataframe[feature_name]
        ):
            raise TypeError(
                f"'{feature_name}' is not a categorical feature."
            )

    @staticmethod
    def _get_cardinality(
        unique_values: int,
    ) -> str:

        if unique_values <= 10:
            return "Low"

        if unique_values <= 100:
            return "Medium"

        return "High"