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

        print("Fitting missing value imputer...")

        print("Selecting numerical and categorical columns...")
        dtypes = dataframe.dtypes
        numerical_columns = dtypes[dtypes.map(pd.api.types.is_numeric_dtype)].index
        categorical_columns = dtypes[~dtypes.map(pd.api.types.is_numeric_dtype)].index

        # ------------------------------------------------------
        # Only process numerical columns that contain NaN
        # ------------------------------------------------------

        print("Finding numerical columns with missing values...")
        numerical_missing_columns = pd.Index([
            col for col in numerical_columns
            if dataframe[col].hasnans
        ])

        print(
            f"Numerical columns with missing values : "
            f"{len(numerical_missing_columns)}"
        )

        print(f"Calculating {self.numerical_strategy} for numerical columns...")
        if self.numerical_strategy == "mean":
            values = pd.Series({
                col: dataframe[col].mean()
                for col in numerical_missing_columns
            })
        else:
            values = pd.Series({
                col: dataframe[col].median()
                for col in numerical_missing_columns
            })

        self.numerical_fill_values = (
            values
            .dropna()
            .to_dict()
        )

        # ------------------------------------------------------
        # Categorical columns
        # ------------------------------------------------------

        print("Finding categorical columns with missing values...")
        categorical_missing_columns = pd.Index([
            col for col in categorical_columns
            if dataframe[col].hasnans
        ])

        print(
            f"Categorical columns with missing values : "
            f"{len(categorical_missing_columns)}"
        )

        print(f"Applying '{self.categorical_strategy}' strategy for categorical columns...")
        if self.categorical_strategy == "missing":

            self.categorical_fill_values = {
                column: "Missing"
                for column in categorical_missing_columns
            }

        else:

            for column in categorical_missing_columns:

                mode = dataframe[column].mode()

                value = (
                    mode.iloc[0]
                    if not mode.empty
                    else "Missing"
                )

                self.categorical_fill_values[column] = value

        print(
            "Missing value imputer fitted."
        )

        return self

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        print("Transforming missing values...")

        # Make one copy because we don't want to modify
        # the original train/test dataframe.

        transformed = dataframe.copy()

        # ------------------------------------------------------
        # Numerical columns
        # ------------------------------------------------------

        if self.numerical_fill_values:

            transformed[
                list(self.numerical_fill_values.keys())
            ] = transformed[
                list(self.numerical_fill_values.keys())
            ].fillna(
                self.numerical_fill_values
            )

        # ------------------------------------------------------
        # Categorical columns
        # ------------------------------------------------------

        if self.categorical_fill_values:

            transformed[
                list(self.categorical_fill_values.keys())
            ] = transformed[
                list(self.categorical_fill_values.keys())
            ].fillna(
                self.categorical_fill_values
            )

        print("Missing value transformation complete.")

        return transformed

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        self.fit(dataframe)

        return self.transform(dataframe)