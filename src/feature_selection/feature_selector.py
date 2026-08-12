import pandas as pd

from src.feature_selection.models import (
    FeatureDecision,
    FeatureSelectionReport,
)


class FeatureSelector:
    """
    Rule-based feature selector.

    Version 1 supports:
    - Identifier detection
    - Constant feature detection
    """

    IDENTIFIER_COLUMNS = {
        "TransactionID",
    }

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> FeatureSelectionReport:

        decisions: list[FeatureDecision] = []

        for column in dataframe.columns:

            # -----------------------------
            # Identifier columns
            # -----------------------------
            if column in self.IDENTIFIER_COLUMNS:

                decisions.append(
                    FeatureDecision(
                        feature_name=column,
                        keep=False,
                        reason="Identifier column",
                    )
                )

                continue

            # -----------------------------
            # Constant columns
            # -----------------------------
            if dataframe[column].nunique(dropna=False) <= 1:

                decisions.append(
                    FeatureDecision(
                        feature_name=column,
                        keep=False,
                        reason="Constant feature",
                    )
                )

                continue

            # -----------------------------
            # Default
            # -----------------------------
            decisions.append(
                FeatureDecision(
                    feature_name=column,
                    keep=True,
                    reason="No exclusion rule matched",
                )
            )

        return FeatureSelectionReport(decisions)