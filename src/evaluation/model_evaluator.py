import pandas as pd
from sklearn.base import BaseEstimator

from src.evaluation.confusion_matrix_builder import (
    ConfusionMatrixBuilder,
)
from src.evaluation.metrics_calculator import (
    MetricsCalculator,
)
from src.evaluation.models import (
    EvaluationResult,
)


class ModelEvaluator:

    def __init__(self) -> None:

        self.metrics_calculator = MetricsCalculator()
        self.confusion_matrix_builder = (
            ConfusionMatrixBuilder()
        )

    def evaluate(
        self,
        model: BaseEstimator,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> EvaluationResult:

        y_prediction = model.predict(
            x_test,
        )

        y_probability = model.predict_proba(
            x_test,
        )[:, 1]

        metrics = self.metrics_calculator.calculate(
            y_true=y_test,
            y_pred=y_prediction,
            y_probability=y_probability,
        )

        confusion_matrix = (
            self.confusion_matrix_builder.build(
                y_true=y_test,
                y_pred=y_prediction,
            )
        )

        return EvaluationResult(
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            roc_auc=metrics["roc_auc"],
            confusion_matrix=confusion_matrix,
        )