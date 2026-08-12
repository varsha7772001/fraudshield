import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class MetricsCalculator:

    def calculate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_probability: pd.Series,
    ) -> dict:

        return {
            "accuracy": accuracy_score(
                y_true,
                y_pred,
            ),
            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "roc_auc": roc_auc_score(
                y_true,
                y_probability,
            ),
        }