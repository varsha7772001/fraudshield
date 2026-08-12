from typing import Iterable

import pandas as pd


class FeatureDropper:

    def __init__(
        self,
        columns: Iterable[str] | None = None,
    ) -> None:

        self.columns = list(columns or [])

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> "FeatureDropper":

        return self

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        return dataframe.drop(
            columns=self.columns,
            errors="ignore",
        )

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        self.fit(dataframe)

        return self.transform(dataframe)