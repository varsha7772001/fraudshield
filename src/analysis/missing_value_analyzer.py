from dataclasses import dataclass

import pandas as pd


@dataclass
class MissingValueSummary:
    feature_name: str

    total_rows: int

    missing_count: int

    missing_percentage: float

    non_missing_count: int

    non_missing_percentage: float

    data_type: str


class MissingValueAnalyzer:

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        self.dataframe = dataframe

    def analyze(
        self,
        feature_name: str,
    ) -> MissingValueSummary:

        self._validate_feature(feature_name)

        total_rows = len(self.dataframe)

        missing_count = int(
            self.dataframe[feature_name].isna().sum()
        )

        non_missing_count = total_rows - missing_count

        missing_percentage = round(
            (missing_count / total_rows) * 100,
            2,
        )

        non_missing_percentage = round(
            (non_missing_count / total_rows) * 100,
            2,
        )

        return MissingValueSummary(
            feature_name=feature_name,
            total_rows=total_rows,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            non_missing_count=non_missing_count,
            non_missing_percentage=non_missing_percentage,
            data_type=str(
                self.dataframe[feature_name].dtype
            ),
        )

    def analyze_all(
        self,
    ) -> list[MissingValueSummary]:

        summaries = []

        for feature_name in self.dataframe.columns:
            summaries.append(
                self.analyze(feature_name)
            )

        summaries.sort(
            key=lambda x: x.missing_percentage,
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