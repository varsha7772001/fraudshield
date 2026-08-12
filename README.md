# FraudShield 🛡️

FraudShield is a comprehensive, end-to-end Machine Learning pipeline designed for detecting fraudulent transactions. It provides a robust, modular architecture for data processing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## 🚀 Overview

The project is structured into distinct phases, each handling a specific part of the machine learning lifecycle:

1. **Data Loading**: Ingests transaction and identity datasets.
2. **Dataset Validation**: Ensures data integrity, checking for transaction ID uniqueness and valid relationships.
3. **Data Merging**: Combines transaction and identity data into a unified dataset.
4. **Exploratory Data Analysis (EDA)**: Offers deep insights into the data:
   - Target and Dataset profiling
   - Numerical and Categorical feature analysis
   - Missing value analysis
   - Feature correlation
5. **Feature Selection**: Identifies and retains the most predictive features while dropping redundant or noisy ones.
6. **Data Preprocessing**: 
   - Train/Test splitting to prevent data leakage.
   - Missing value imputation (e.g., median for numerical, 'missing' category for categorical).
   - Robust categorical encoding.
7. **Model Training**: Trains a baseline `RandomForestClassifier` for robust fraud detection.
8. **Evaluation & Explainability**: Evaluates model performance and provides feature importance analysis to explain model decisions.

## 📁 Project Structure

- `src/main.py`: The main entry point orchestrating the entire pipeline.
- `src/data/`: Modules for data loading and merging.
- `src/analysis/`: Comprehensive tools for EDA, data profiling, and feature analysis.
- `src/preprocessing/`: Handlers for missing values, encoding, and data splitting.
- `src/feature_selection/`: Logic for selecting optimal features.
- `src/training/`: Model training wrappers and utilities.
- `src/evaluation/`: Metrics calculation and model evaluation.
- `src/explainability/`: Tools for understanding model predictions (feature importance).
- `src/models/` & `src/visualization/`: Helper structures and visualization tools.

## 🛠️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/varsha7772001/fraudshield.git
   cd fraudshield
   ```

2. Create a virtual environment and activate it (recommended):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Kaggle API Setup** (for automatic data download):
   - Create a Kaggle account and download your API token (`kaggle.json`).
   - Place `kaggle.json` in `~/.kaggle/kaggle.json` (or `C:\Users\<YourUsername>\.kaggle\kaggle.json` on Windows).
   - Ensure the file has appropriate read permissions.
   - If the `data/raw` folder is empty, the pipeline will automatically download the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data).

## 💻 Usage

Run the main pipeline:

```bash
python -m src.main
```

Ensure your datasets (e.g., transaction and identity data) are placed in the appropriate `data/` directory before running the pipeline.
