import matplotlib.pyplot as plt
import pandas as pd


class NumericalFeatureVisualizer:

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def plot_histogram(
        self,
        feature_name: str,
        bins: int = 50,
    ) -> None:

        self._validate_feature(feature_name)

        plt.figure(figsize=(10, 6))
        plt.hist(
            self.dataframe[feature_name].dropna(),
            bins=bins,
        )

        plt.title(f"{feature_name} Distribution")
        plt.xlabel(feature_name)
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def plot_boxplot(
        self,
        feature_name: str,
    ) -> None:

        self._validate_feature(feature_name)

        plt.figure(figsize=(8, 5))

        plt.boxplot(
            self.dataframe[feature_name].dropna(),
            vert=False,
        )

        plt.title(f"{feature_name} Box Plot")
        plt.xlabel(feature_name)

        plt.tight_layout()
        plt.show()

    def plot_histogram_by_target(
        self,
        feature_name: str,
        target_column: str = "isFraud",
        bins: int = 50,
    ) -> None:

        self._validate_feature(feature_name)

        fraud = self.dataframe[
            self.dataframe[target_column] == 1
        ][feature_name].dropna()

        legitimate = self.dataframe[
            self.dataframe[target_column] == 0
        ][feature_name].dropna()

        plt.figure(figsize=(10, 6))

        plt.hist(
            legitimate,
            bins=bins,
            alpha=0.5,
            label="Legitimate",
        )

        plt.hist(
            fraud,
            bins=bins,
            alpha=0.5,
            label="Fraud",
        )

        plt.title(f"{feature_name} by Target")
        plt.xlabel(feature_name)
        plt.ylabel("Frequency")

        plt.legend()

        plt.tight_layout()
        plt.show()

    def _validate_feature(
        self,
        feature_name: str,
    ) -> None:

        if feature_name not in self.dataframe.columns:
            raise ValueError(
                f"{feature_name} does not exist."
            )

        if not pd.api.types.is_numeric_dtype(
            self.dataframe[feature_name]
        ):
            raise TypeError(
                f"{feature_name} is not numeric."
            )