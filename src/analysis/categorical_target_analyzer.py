from dataclasses import dataclass

import pandas as pd


@dataclass
class CategoryStatistics:
    category: str

    total_count: int

    fraud_count: int

    legitimate_count: int

    fraud_rate: float

    legitimate_rate: float


@dataclass
class CategoricalTargetSummary:
    feature_name: str

    total_categories: int

    categories: list[CategoryStatistics]


class CategoricalTargetAnalyzer:

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
    ) -> CategoricalTargetSummary:

        self._validate_feature(feature_name)

        category_statistics = []

        grouped = self.dataframe.groupby(feature_name)

        for category, group in grouped:

            total_count = len(group)

            fraud_count = int(group[self.target_column].sum())

            legitimate_count = total_count - fraud_count

            fraud_rate = round(
                (fraud_count / total_count) * 100,
                2,
            )

            legitimate_rate = round(
                (legitimate_count / total_count) * 100,
                2,
            )

            category_statistics.append(
                CategoryStatistics(
                    category=str(category),
                    total_count=total_count,
                    fraud_count=fraud_count,
                    legitimate_count=legitimate_count,
                    fraud_rate=fraud_rate,
                    legitimate_rate=legitimate_rate,
                )
            )

        category_statistics.sort(
            key=lambda x: x.fraud_rate,
            reverse=True,
        )

        return CategoricalTargetSummary(
            feature_name=feature_name,
            total_categories=len(category_statistics),
            categories=category_statistics,
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

        if self.target_column not in self.dataframe.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found."
            )