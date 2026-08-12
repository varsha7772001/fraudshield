import pandas as pd

from ..models.feature_catalog import FeatureSummary


class FeatureAnalyzer:

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe

    def summarize(self) -> FeatureSummary:

        target = "isFraud"

        identifiers = [
            "TransactionID"
        ]

        numeric_features = (
            self.dataframe
            .select_dtypes(include="number")
            .columns
            .difference([target] + identifiers)
            .tolist()
        )

        categorical_features = (
            self.dataframe
            .select_dtypes(include="object")
            .columns
            .tolist()
        )

        return FeatureSummary(
            target=target,
            identifiers=identifiers,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )