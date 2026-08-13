def transform(
    self,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    print("Transforming missing values...")

    fill_values = {
        **self.numerical_fill_values,
        **self.categorical_fill_values,
    }

    dataframe.fillna(
        value=fill_values,
        inplace=True,
    )

    print("Missing value transformation complete.")

    return dataframe