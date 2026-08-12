from dataclasses import dataclass

import pandas as pd


@dataclass
class FeatureCorrelation:
    feature_name: str
    correlation: float


@dataclass
class FeatureCorrelationSummary:
    source_feature: str
    correlations: list[FeatureCorrelation]


class FeatureCorrelationAnalyzer:

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        self.dataframe = dataframe

    def analyze(
        self,
        feature_name: str,
    ) -> FeatureCorrelationSummary:

        self._validate_feature(feature_name)

        correlations = self.dataframe.corr(
            numeric_only=True
        )[feature_name]

        feature_correlations = []

        for correlated_feature, correlation in correlations.items():

            if correlated_feature == feature_name:
                continue

            if pd.isna(correlation):
                continue

            feature_correlations.append(
                FeatureCorrelation(
                    feature_name=correlated_feature,
                    correlation=round(
                        float(correlation),
                        4,
                    ),
                )
            )

        feature_correlations.sort(
            key=lambda x: abs(x.correlation),
            reverse=True,
        )

        return FeatureCorrelationSummary(
            source_feature=feature_name,
            correlations=feature_correlations,
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
                f"'{feature_name}' is not numerical."
            )