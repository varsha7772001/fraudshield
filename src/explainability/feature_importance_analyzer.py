import pandas as pd
from sklearn.base import BaseEstimator

from src.explainability.models import FeatureImportance


class FeatureImportanceAnalyzer:

    def analyze(
        self,
        model: BaseEstimator,
        feature_names: list[str],
    ) -> list[FeatureImportance]:

        if not hasattr(model, "feature_importances_"):
            raise TypeError(
                "Model does not provide feature_importances_. "
                "Feature importance analysis requires a "
                "tree-based model such as Random Forest."
            )

        importances = model.feature_importances_

        if len(importances) != len(feature_names):
            raise ValueError(
                "Number of feature importances does not match "
                "number of feature names."
            )

        importance_dataframe = pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance": importances,
            }
        )

        importance_dataframe = (
            importance_dataframe
            .sort_values(
                by="importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        results: list[FeatureImportance] = []

        for index, row in importance_dataframe.iterrows():

            results.append(
                FeatureImportance(
                    feature_name=row["feature_name"],
                    importance=float(row["importance"]),
                    rank=index + 1,
                )
            )

        return results