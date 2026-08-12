from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass

class TrainTestSplit:

    x_train: pd.DataFrame
    x_test: pd.DataFrame

    y_train: pd.Series
    y_test: pd.Series


class TrainTestSplitter:

    def __init__(
        self,
        target_column: str = "isFraud",
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> None:

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def split(
        self,
        dataframe: pd.DataFrame,
    ) -> TrainTestSplit:

        x = dataframe.drop(columns=[self.target_column])
        y = dataframe[self.target_column]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.stratify else None,
        )

        return TrainTestSplit(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )