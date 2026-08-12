import os
import zipfile
import subprocess
from pathlib import Path

import pandas as pd


class DataLoader:
    """
    Loads raw IEEE-CIS fraud detection datasets.
    """

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.raw_data_path = self.project_root / "data" / "raw"

    def _download_data_if_missing(self) -> None:
        """
        Checks if the CSV data exists. If not, downloads it from Google Drive.
        """
        target_file = self.raw_data_path / "train_transaction.csv"
        
        if not target_file.exists():
            print("Dataset not found. Attempting to download from Google Drive...")
            
            try:
                # Ensure the directory exists
                self.raw_data_path.mkdir(parents=True, exist_ok=True)
                
                # Download using gdown
                command = [
                    "gdown", "--folder", 
                    "https://drive.google.com/drive/folders/1fEiuykJqHKcxY07l0QOhjgqpc2nLrl1j?usp=sharing", 
                    "-O", str(self.raw_data_path)
                ]
                subprocess.run(command, check=True)
                
                print("Download complete.")
                    
            except subprocess.CalledProcessError as e:
                print(f"Error downloading from Google Drive: {e}")
            except FileNotFoundError:
                print("Error: The 'gdown' command was not found.")
                print("Please install it using 'pip install gdown'.")

    def load_train_transaction(self) -> pd.DataFrame:
        self._download_data_if_missing()
        file_path = self.raw_data_path / "train_transaction.csv"
        return pd.read_csv(file_path)

    def load_train_identity(self) -> pd.DataFrame:
        self._download_data_if_missing()
        file_path = self.raw_data_path / "train_identity.csv"
        return pd.read_csv(file_path)