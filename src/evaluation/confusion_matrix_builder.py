import pandas as pd
from sklearn.metrics import confusion_matrix


class ConfusionMatrixBuilder:

    def build(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
    ):

        return confusion_matrix(
            y_true,
            y_pred,
        )