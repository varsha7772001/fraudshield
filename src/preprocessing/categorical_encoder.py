import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class CategoricalEncoder:

    def __init__(self) -> None:

        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float32,
        )

        self.categorical_columns: list[str] = []

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> "CategoricalEncoder":

        self.categorical_columns = (
            dataframe
            .select_dtypes(
                include=["object", "category"]
            )
            .columns
            .tolist()
        )

        if not self.categorical_columns:
            return self

        self.encoder.fit(
            dataframe[self.categorical_columns]
        )

        return self

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        if not self.categorical_columns:
            return dataframe

        encoded_values = self.encoder.transform(
            dataframe[self.categorical_columns]
        )

        # Avoid deep-copying the entire DataFrame.
        transformed = dataframe.copy(deep=False)

        transformed.loc[
            :,
            self.categorical_columns,
        ] = encoded_values

        return transformed

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        self.fit(dataframe)

        return self.transform(dataframe)