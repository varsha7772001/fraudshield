import pandas as pd


class DatasetProfiler:
    """
    Generates a high-level profile of the dataset.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe

    def profile(self) -> None:

        rows, columns = self.dataframe.shape

        numeric_columns = (
            self.dataframe
            .select_dtypes(include="number")
            .columns
        )

        categorical_columns = (
            self.dataframe
            .select_dtypes(include="object")
            .columns
        )

        print("\n" + "=" * 60)
        print("Dataset Profile")
        print("=" * 60)

        print(f"Rows                 : {rows:,}")
        print(f"Columns              : {columns}")
        print(f"Numeric Columns      : {len(numeric_columns)}")
        print(f"Categorical Columns  : {len(categorical_columns)}")

    def profile_missing_values(self) -> None:
        """
        Generate a summary of missing values in the dataset.
        """

        missing_summary = (
            self.dataframe
            .isnull()
            .sum()
            .sort_values(ascending=False)
        )


        missing_count = self.dataframe.isnull().sum()

        missing_percentage = (
            missing_count / len(self.dataframe) * 100
        ).round(2)

        missing_summary = pd.DataFrame(
            {
                "Missing Count": missing_count,
                "Missing Percentage": missing_percentage,
            }
        )

        missing_summary = missing_summary[
            missing_summary["Missing Count"] > 0
        ].sort_values(
            by="Missing Percentage",
            ascending=False,
        )

        print("\n" + "=" * 60)
        print("Missing Values Summary") 
        print("=" * 60)
        print(missing_summary)  

    def profile_data_types(self) -> None:
        """
        Generate a summary of column data types.
        """

        dtype_summary = (
            self.dataframe
            .dtypes
            .value_counts()
        )

        print("\n" + "=" * 60)
        print("Data Type Profile")
        print("=" * 60)

        print(dtype_summary)    