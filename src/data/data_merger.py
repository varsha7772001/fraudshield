import pandas as pd


class DataMerger:
    """
    Merges validated datasets into a single dataset
    for downstream preprocessing and model training.
    """

    def __init__(
        self,
        transaction_df: pd.DataFrame,
        identity_df: pd.DataFrame,
    ) -> None:

        self.transaction_df = transaction_df
        self.identity_df = identity_df

    def merge(self) -> pd.DataFrame:
        """
        Merge transaction and identity datasets.

        Returns:
            pd.DataFrame: Combined dataset.
        """

        merged_df = pd.merge(
            left=self.transaction_df,
            right=self.identity_df,
            on="TransactionID",
            how="left",
            validate="one_to_one",
        )

        return merged_df