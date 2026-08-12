from typing import Literal

import pandas as pd


class MissingValueImputer:

    def __init__(
        self,
        numerical_strategy: Literal["mean", "median"] = "median",
        categorical_strategy: Literal["mode", "missing"] = "missing",
    ) -> None:

        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy

        self.numerical_fill_values: dict[str, float] = {}
        self.categorical_fill_values: dict[str, str] = {}

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> "MissingValueImputer":

        numerical_columns = dataframe.select_dtypes(
            include=["number"]
        ).columns

        categorical_columns = dataframe.select_dtypes(
            exclude=["number"]
        ).columns

        for column in numerical_columns:

            if self.numerical_strategy == "mean":
                value = dataframe[column].mean()

            else:
                value = dataframe[column].median()

            self.numerical_fill_values[column] = value

        for column in categorical_columns:

            if self.categorical_strategy == "mode":

                mode = dataframe[column].mode()

                value = (
                    mode.iloc[0]
                    if not mode.empty
                    else "Missing"
                )

            else:

                value = "Missing"

            self.categorical_fill_values[column] = value

        return self

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        transformed = dataframe.copy()

        for (
            column,
            value,
        ) in self.numerical_fill_values.items():

            transformed[column] = transformed[column].fillna(
                value
            )

        for (
            column,
            value,
        ) in self.categorical_fill_values.items():

            transformed[column] = transformed[column].fillna(
                value
            )

        return transformed

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        self.fit(dataframe)

        return self.transform(dataframe)