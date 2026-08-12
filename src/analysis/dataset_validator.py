import pandas as pd


class DatasetValidator:
    """
    Performs structural validation on the fraud datasets.

    Responsibilities:
    - Validate assumptions before merging.
    - Check dataset integrity.
    - Report structural issues.

    This class does NOT:
    - Load data
    - Merge data
    - Clean data
    """

    def __init__(
        self,
        transaction_df: pd.DataFrame,
        identity_df: pd.DataFrame,
    ) -> None:

        self.transaction_df = transaction_df
        self.identity_df = identity_df

    def validate_transaction_id_uniqueness(self) -> None:
        """
        Verify that TransactionID is unique
        in both datasets.
        """

        transaction_duplicates = (
            self.transaction_df["TransactionID"]
            .duplicated()
            .sum()
        )

        identity_duplicates = (
            self.identity_df["TransactionID"]
            .duplicated()
            .sum()
        )

        print("=" * 60)
        print("TransactionID Validation")
        print("=" * 60)

        print(
            f"Duplicate Transaction IDs (Transactions): {transaction_duplicates}"
        )

        print(
            f"Duplicate Transaction IDs (Identity): {identity_duplicates}"
        )

        if transaction_duplicates == 0:
            print("✓ Transaction table has unique TransactionID values.")
        else:
            print("✗ Transaction table contains duplicate TransactionID values.")

        if identity_duplicates == 0:
            print("✓ Identity table has unique TransactionID values.")
        else:
            print("✗ Identity table contains duplicate TransactionID values.")

    def validate_transaction_relationship(self) -> None:
        """
        Validate the relationship between the
        transaction and identity datasets.
        """

        transaction_ids = set(self.transaction_df["TransactionID"])

        identity_ids = set(self.identity_df["TransactionID"])

        matching_ids = transaction_ids.intersection(identity_ids)

        orphan_identity_ids = identity_ids - transaction_ids

        transactions_without_identity = (
            len(transaction_ids) - len(matching_ids)
        )

        print("\n" + "=" * 60)
        print("Transaction Relationship Validation")
        print("=" * 60)

        print(f"Transactions                : {len(transaction_ids):,}")
        print(f"Identity Records            : {len(identity_ids):,}")
        print(f"Matching Transaction IDs    : {len(matching_ids):,}")
        print(f"Transactions Without Identity: {transactions_without_identity:,}")
        print(f"Orphan Identity Records     : {len(orphan_identity_ids):,}")