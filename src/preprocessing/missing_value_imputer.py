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

    # ==========================================================
    # FIT
    # ==========================================================

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> "MissingValueImputer":

        print("Fitting missing value imputer...")

        # ------------------------------------------------------
        # Identify numerical and categorical columns
        # ------------------------------------------------------

        print(
            "Selecting numerical and categorical columns..."
        )

        dtypes = dataframe.dtypes

        numerical_columns = dtypes[
            dtypes.map(
                pd.api.types.is_numeric_dtype
            )
        ].index

        categorical_columns = dtypes[
            ~dtypes.map(
                pd.api.types.is_numeric_dtype
            )
        ].index

        # ------------------------------------------------------
        # Numerical columns with missing values
        # ------------------------------------------------------

        print(
            "Finding numerical columns with "
            "missing values..."
        )

        numerical_missing_columns = pd.Index(
            [
                column
                for column in numerical_columns
                if dataframe[column].hasnans
            ]
        )

        print(
            "Numerical columns with missing values : "
            f"{len(numerical_missing_columns)}"
        )

        # ------------------------------------------------------
        # Calculate numerical fill values
        # ------------------------------------------------------

        print(
            f"Calculating {self.numerical_strategy} "
            "for numerical columns..."
        )

        if len(numerical_missing_columns) > 0:

            if self.numerical_strategy == "mean":

                values = dataframe[
                    numerical_missing_columns
                ].mean()

            else:

                values = dataframe[
                    numerical_missing_columns
                ].median()

            self.numerical_fill_values = (
                values
                .dropna()
                .to_dict()
            )

        else:

            self.numerical_fill_values = {}

        # ------------------------------------------------------
        # Categorical columns with missing values
        # ------------------------------------------------------

        print(
            "Finding categorical columns with "
            "missing values..."
        )

        categorical_missing_columns = pd.Index(
            [
                column
                for column in categorical_columns
                if dataframe[column].hasnans
            ]
        )

        print(
            "Categorical columns with missing values : "
            f"{len(categorical_missing_columns)}"
        )

        # ------------------------------------------------------
        # Calculate categorical fill values
        # ------------------------------------------------------

        print(
            f"Applying '{self.categorical_strategy}' "
            "strategy for categorical columns..."
        )

        if self.categorical_strategy == "missing":

            self.categorical_fill_values = {
                column: "Missing"
                for column in categorical_missing_columns
            }

        else:

            self.categorical_fill_values = {}

            for column in categorical_missing_columns:

                mode = dataframe[column].mode()

                value = (
                    mode.iloc[0]
                    if not mode.empty
                    else "Missing"
                )

                self.categorical_fill_values[
                    column
                ] = value

        print(
            "Missing value imputer fitted."
        )

        return self

    # ==========================================================
    # TRANSFORM
    # ==========================================================

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        print(
            "Transforming missing values..."
        )

        # ------------------------------------------------------
        # Combine numerical and categorical fill values
        # ------------------------------------------------------

        fill_values = {
            **self.numerical_fill_values,
            **self.categorical_fill_values,
        }

        # ------------------------------------------------------
        # IMPORTANT:
        #
        # Modify the supplied dataframe in-place.
        #
        # This avoids creating another complete copy of
        # 472,432 x 432 data.
        # ------------------------------------------------------

        if fill_values:

            dataframe.fillna(
                value=fill_values,
                inplace=True,
            )

        print(
            "Missing value transformation complete."
        )

        return dataframe

    # ==========================================================
    # FIT TRANSFORM
    # ==========================================================

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        self.fit(dataframe)

        return self.transform(dataframe)