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
        Checks if the CSV data exists. If not, downloads it using the Kaggle API.
        """
        target_file = self.raw_data_path / "train_transaction.csv"
        
        if not target_file.exists():
            print("Dataset not found. Attempting to download from Kaggle...")
            
            try:
                # Ensure the directory exists
                self.raw_data_path.mkdir(parents=True, exist_ok=True)
                
                # Download using Kaggle CLI
                command = [
                    "kaggle", "competitions", "download", 
                    "-c", "ieee-fraud-detection", 
                    "-p", str(self.raw_data_path)
                ]
                subprocess.run(command, check=True)
                
                zip_path = self.raw_data_path / "ieee-fraud-detection.zip"
                
                if zip_path.exists():
                    print("Download complete. Extracting files...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(self.raw_data_path)
                    
                    print("Extraction complete. Cleaning up zip file...")
                    os.remove(zip_path)
                else:
                    print("Error: Zip file was not downloaded properly.")
                    
            except subprocess.CalledProcessError as e:
                print(f"Error downloading from Kaggle: {e}")
                print("Make sure you have configured your kaggle.json token in ~/.kaggle/kaggle.json")
            except FileNotFoundError:
                print("Error: The 'kaggle' command was not found.")
                print("Please install it using 'pip install kaggle' and configure your API token.")

    def load_train_transaction(self) -> pd.DataFrame:
        self._download_data_if_missing()
        file_path = self.raw_data_path / "train_transaction.csv"
        return pd.read_csv(file_path)

    def load_train_identity(self) -> pd.DataFrame:
        self._download_data_if_missing()
        file_path = self.raw_data_path / "train_identity.csv"
        return pd.read_csv(file_path)