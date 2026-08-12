from pathlib import Path

import pandas as pd


class DataLoader:
    """
    Loads raw IEEE-CIS fraud detection datasets.
    """

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.raw_data_path = self.project_root / "data" / "raw"

    def load_train_transaction(self) -> pd.DataFrame:
        file_path = self.raw_data_path / "train_transaction.csv"
        return pd.read_csv(file_path)

    def load_train_identity(self) -> pd.DataFrame:
        file_path = self.raw_data_path / "train_identity.csv"
        return pd.read_csv(file_path)