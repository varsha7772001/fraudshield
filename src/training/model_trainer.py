import time

import pandas as pd
from sklearn.base import BaseEstimator

from src.training.training_result import (
    TrainingResult,
)


class ModelTrainer:

    def __init__(
        self,
        model: BaseEstimator,
    ) -> None:

        self.model = model

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> TrainingResult:

        start_time = time.perf_counter()

        self.model.fit(
            x_train,
            y_train,
        )

        end_time = time.perf_counter()

        return TrainingResult(
            model=self.model,
            training_time=end_time - start_time,
        )