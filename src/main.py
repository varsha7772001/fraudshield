# ==========================================================
# Data
# ==========================================================
from src.data.data_loader import DataLoader
from src.data.data_merger import DataMerger


# ==========================================================
# Validation
# ==========================================================
from src.analysis.dataset_validator import DatasetValidator


# ==========================================================
# Analysis
# ==========================================================
from src.analysis.target_analyzer import TargetAnalyzer
from src.analysis.dataset_profiler import DatasetProfiler
from src.analysis.numerical_feature_analyzer import (
    NumericalFeatureAnalyzer,
)
from src.analysis.categorical_feature_analyzer import (
    CategoricalFeatureAnalyzer,
)
from src.analysis.categorical_target_analyzer import (
    CategoricalTargetAnalyzer,
)
from src.analysis.numerical_target_analyzer import (
    NumericalTargetAnalyzer,
)
from src.analysis.missing_value_analyzer import (
    MissingValueAnalyzer,
)
from src.analysis.missing_value_target_analyzer import (
    MissingValueTargetAnalyzer,
)
from src.analysis.feature_correlation_analyzer import (
    FeatureCorrelationAnalyzer,
)


# ==========================================================
# Models
# ==========================================================
from src.models.feature_catalog import FeatureCatalogBuilder


# ==========================================================
# Visualization
# ==========================================================
from src.visualization.numerical_feature_visualizer import (
    NumericalFeatureVisualizer,
)


# ==========================================================
# Feature Selection
# ==========================================================
from src.feature_selection.feature_selector import (
    FeatureSelector,
)


# ==========================================================
# Preprocessing
# ==========================================================
from src.preprocessing.feature_dropper import (
    FeatureDropper,
)
from src.preprocessing.missing_value_imputer import (
    MissingValueImputer,
)
from src.preprocessing.train_test_splitter import (
    TrainTestSplitter,
)
from src.preprocessing.categorical_encoder import (
    CategoricalEncoder,
)


# ==========================================================
# Training
# ==========================================================
from sklearn.ensemble import RandomForestClassifier

from src.training.model_trainer import (
    ModelTrainer,
)


# ==========================================================
# Evaluation
# ==========================================================
from src.evaluation.model_evaluator import (
    ModelEvaluator,
)


# ==========================================================
# Explainability
# ==========================================================
from src.explainability.feature_importance_analyzer import (
    FeatureImportanceAnalyzer,
)
from src.explainability.feature_importance_reporter import (
    FeatureImportanceReporter,
)


def main() -> None:

    # ==========================================================
    # PHASE 1 - Data Loading
    # ==========================================================

    loader = DataLoader()

    transaction_df = loader.load_train_transaction()
    identity_df = loader.load_train_identity()


    # ==========================================================
    # PHASE 2 - Dataset Validation
    # ==========================================================

    validator = DatasetValidator(
        transaction_df,
        identity_df,
    )

    validator.validate_transaction_id_uniqueness()
    validator.validate_transaction_relationship()


    # ==========================================================
    # PHASE 3 - Merge Dataset
    # ==========================================================

    merger = DataMerger(
        transaction_df,
        identity_df,
    )

    merged_df = merger.merge()

    print("\n" + "=" * 60)
    print("Merged Dataset")
    print("=" * 60)

    print(f"Shape: {merged_df.shape}")


    # ==========================================================
    # PHASE 4 - Exploratory Data Analysis
    # ==========================================================

    # ----------------------------------------------------------
    # Target Analyzer
    # ----------------------------------------------------------

    # target_analyzer = TargetAnalyzer(merged_df)
    # target_analyzer.analyze()


    # ----------------------------------------------------------
    # Dataset Profiler
    # ----------------------------------------------------------

    # profiler = DatasetProfiler(merged_df)

    # profiler.profile()
    # profiler.profile_missing_values()
    # profiler.profile_data_types()


    # ----------------------------------------------------------
    # Feature Catalog
    # ----------------------------------------------------------

    # print("\n" + "=" * 60)
    # print("Feature Catalog")
    # print("=" * 60)

    # catalog = FeatureCatalogBuilder(
    #     merged_df
    # ).build()

    # print(f"Total Columns : {len(catalog.features)}")
    # print(f"Target        : {catalog.target}")
    # print(f"Identifiers   : {catalog.identifiers}")

    # print("\nFirst 10 Features\n")

    # for feature in catalog.features[:10]:
    #     print(feature)


    # ----------------------------------------------------------
    # Numerical Feature Analyzer
    # ----------------------------------------------------------

    # analyzer = NumericalFeatureAnalyzer(
    #     merged_df
    # )

    # summary = analyzer.analyze(
    #     "TransactionAmt"
    # )

    # print("\n" + "=" * 60)
    # print("Transaction Amount Summary")
    # print("=" * 60)

    # print(
    #     f"Feature Name          : "
    #     f"{summary.feature_name}"
    # )

    # print(
    #     f"Data Type             : "
    #     f"{summary.data_type}"
    # )

    # print(
    #     f"Total Count           : "
    #     f"{summary.total_count}"
    # )

    # print(
    #     f"Missing Count         : "
    #     f"{summary.missing_count}"
    # )

    # print(
    #     f"Missing Percentage    : "
    #     f"{summary.missing_percentage}%"
    # )

    # print(
    #     f"Minimum               : "
    #     f"{summary.minimum}"
    # )

    # print(
    #     f"Maximum               : "
    #     f"{summary.maximum}"
    # )

    # print(
    #     f"Mean                  : "
    #     f"{summary.mean}"
    # )

    # print(
    #     f"Median                : "
    #     f"{summary.median}"
    # )

    # print(
    #     f"Standard Deviation    : "
    #     f"{summary.standard_deviation}"
    # )

    # print(
    #     f"Skewness              : "
    #     f"{summary.skewness}"
    # )

    # print(
    #     f"Kurtosis              : "
    #     f"{summary.kurtosis}"
    # )

    # print(
    #     f"Unique Values         : "
    #     f"{summary.unique_values}"
    # )


    # ----------------------------------------------------------
    # Numerical Feature Visualizer
    # ----------------------------------------------------------

    # visualizer = NumericalFeatureVisualizer(
    #     merged_df
    # )

    # visualizer.plot_histogram(
    #     "TransactionAmt"
    # )

    # visualizer.plot_boxplot(
    #     "TransactionAmt"
    # )

    # visualizer.plot_histogram_by_target(
    #     "TransactionAmt"
    # )


    # ----------------------------------------------------------
    # Categorical Feature Analyzer
    # ----------------------------------------------------------

    # analyzer = CategoricalFeatureAnalyzer(
    #     merged_df
    # )

    # summary = analyzer.analyze(
    #     "ProductCD"
    # )

    # print("\n" + "=" * 60)
    # print("ProductCD Summary")
    # print("=" * 60)

    # print(
    #     f"Feature Name             : "
    #     f"{summary.feature_name}"
    # )

    # print(
    #     f"Data Type                : "
    #     f"{summary.data_type}"
    # )

    # print(
    #     f"Total Count              : "
    #     f"{summary.total_count}"
    # )

    # print(
    #     f"Missing Count            : "
    #     f"{summary.missing_count}"
    # )

    # print(
    #     f"Missing Percentage       : "
    #     f"{summary.missing_percentage}%"
    # )

    # print(
    #     f"Unique Values            : "
    #     f"{summary.unique_values}"
    # )

    # print(
    #     f"Most Frequent Value      : "
    #     f"{summary.most_frequent_value}"
    # )

    # print(
    #     f"Most Frequent Count      : "
    #     f"{summary.most_frequent_count}"
    # )

    # print(
    #     f"Most Frequent Percentage : "
    #     f"{summary.most_frequent_percentage}%"
    # )

    # print(
    #     f"Cardinality              : "
    #     f"{summary.cardinality}"
    # )


    # ----------------------------------------------------------
    # Categorical Target Analyzer
    # ----------------------------------------------------------

    # analyzer = CategoricalTargetAnalyzer(
    #     merged_df
    # )

    # summary = analyzer.analyze(
    #     "ProductCD"
    # )

    # print("\n" + "=" * 70)
    # print(
    #     f"{summary.feature_name} Fraud Analysis"
    # )
    # print("=" * 70)

    # for category in summary.categories:

    #     print(
    #         f"Category              : "
    #         f"{category.category}"
    #     )

    #     print(
    #         f"Total Transactions    : "
    #         f"{category.total_count}"
    #     )

    #     print(
    #         f"Fraud Transactions    : "
    #         f"{category.fraud_count}"
    #     )

    #     print(
    #         f"Legitimate            : "
    #         f"{category.legitimate_count}"
    #     )

    #     print(
    #         f"Fraud Rate            : "
    #         f"{category.fraud_rate}%"
    #     )

    #     print(
    #         f"Legitimate Rate       : "
    #         f"{category.legitimate_rate}%"
    #     )

    #     print("-" * 70)


    # ----------------------------------------------------------
    # Numerical Target Analyzer
    # ----------------------------------------------------------

    # analyzer = NumericalTargetAnalyzer(
    #     merged_df
    # )

    # summary = analyzer.analyze(
    #     "TransactionAmt"
    # )

    # print("\n" + "=" * 70)
    # print(
    #     f"{summary.feature_name} vs Target"
    # )
    # print("=" * 70)

    # print(
    #     f"Fraud Transactions        : "
    #     f"{summary.fraud_count}"
    # )

    # print(
    #     f"Legitimate Transactions   : "
    #     f"{summary.legitimate_count}"
    # )

    # print()

    # print(
    #     f"Fraud Mean               : "
    #     f"{summary.fraud_mean}"
    # )

    # print(
    #     f"Legitimate Mean          : "
    #     f"{summary.legitimate_mean}"
    # )

    # print()

    # print(
    #     f"Fraud Median             : "
    #     f"{summary.fraud_median}"
    # )

    # print(
    #     f"Legitimate Median        : "
    #     f"{summary.legitimate_median}"
    # )

    # print()

    # print(
    #     f"Fraud Minimum            : "
    #     f"{summary.fraud_minimum}"
    # )

    # print(
    #     f"Legitimate Minimum       : "
    #     f"{summary.legitimate_minimum}"
    # )

    # print()

    # print(
    #     f"Fraud Maximum            : "
    #     f"{summary.fraud_maximum}"
    # )

    # print(
    #     f"Legitimate Maximum       : "
    #     f"{summary.legitimate_maximum}"
    # )

    # print()

    # print(
    #     f"Fraud Std Dev            : "
    #     f"{summary.fraud_standard_deviation}"
    # )

    # print(
    #     f"Legitimate Std Dev       : "
    #     f"{summary.legitimate_standard_deviation}"
    # )


    # ----------------------------------------------------------
    # Missing Value Analyzer
    # ----------------------------------------------------------

    # analyzer = MissingValueAnalyzer(
    #     merged_df
    # )

    # summaries = analyzer.analyze_all()

    # print("\n" + "=" * 90)
    # print(
    #     "Top 20 Features With Highest Missing Values"
    # )
    # print("=" * 90)

    # for summary in summaries[:20]:

    #     print(
    #         f"{summary.feature_name:<20}"
    #         f"{summary.missing_percentage:>8}%"
    #         f" ({summary.missing_count:,})"
    #     )


    # ----------------------------------------------------------
    # Missing Value Target Analyzer
    # ----------------------------------------------------------

    # analyzer = MissingValueTargetAnalyzer(
    #     merged_df
    # )

    # summaries = analyzer.analyze_all()

    # print("\n" + "=" * 120)
    # print(
    #     "Top 20 Features With Highest Fraud Rate Difference"
    # )
    # print("=" * 120)

    # for summary in summaries[:20]:

    #     print(
    #         f"{summary.feature_name:<12}"
    #         f"Missing: {summary.missing_count:>8,} "
    #         f"({summary.missing_percentage:>6}%)  "
    #         f"Fraud: {summary.missing_fraud_rate:>6}%   |   "
    #         f"Available: {summary.available_count:>8,} "
    #         f"({summary.available_percentage:>6}%)  "
    #         f"Fraud: {summary.available_fraud_rate:>6}%   |   "
    #         f"Difference: {summary.fraud_rate_difference:>6}%"
    #     )


    # ----------------------------------------------------------
    # Feature Correlation Analyzer
    # ----------------------------------------------------------

    # analyzer = FeatureCorrelationAnalyzer(
    #     merged_df
    # )

    # summary = analyzer.analyze(
    #     "TransactionAmt"
    # )

    # print("\n" + "=" * 80)
    # print(
    #     f"Top Correlated Features with "
    #     f"{summary.source_feature}"
    # )
    # print("=" * 80)

    # for correlation in summary.correlations[:20]:

    #     print(
    #         f"{correlation.feature_name:<20}"
    #         f"{correlation.correlation:>8}"
    #     )


    # ==========================================================
    # PHASE 5 - Feature Selection
    # ==========================================================

    selector = FeatureSelector()

    report = selector.fit(
        merged_df
    )

    print("\n" + "=" * 70)
    print("Feature Selection")
    print("=" * 70)

    print(
        f"Columns to Drop : "
        f"{len(report.columns_to_drop())}"
    )

    print(
        report.columns_to_drop()
    )

    print()

    print(
        f"Columns to Keep : "
        f"{len(report.columns_to_keep())}"
    )


    # ==========================================================
    # PHASE 6 - Feature Dropping
    # ==========================================================

    dropper = FeatureDropper(
        columns=report.columns_to_drop(),
    )

    processed_df = dropper.fit_transform(
        merged_df,
    )


    # ==========================================================
    # PHASE 7 - Train/Test Split
    # ==========================================================

    splitter = TrainTestSplitter()

    split = splitter.split(
        processed_df,
    )

    print("\n" + "=" * 70)
    print("Train/Test Split")
    print("=" * 70)

    print(
        f"X Train : "
        f"{split.x_train.shape}"
    )

    print(
        f"X Test  : "
        f"{split.x_test.shape}"
    )

    print(
        f"Y Train : "
        f"{split.y_train.shape}"
    )

    print(
        f"Y Test  : "
        f"{split.y_test.shape}"
    )


    # ==========================================================
    # PHASE 8 - Missing Value Imputation
    # ==========================================================

    imputer = MissingValueImputer(
        numerical_strategy="median",
        categorical_strategy="missing",
    )

    # IMPORTANT:
    # Fit ONLY on training data.
    # This prevents test-data leakage.

    imputer.fit(
        split.x_train,
    )

    x_train = imputer.transform(
        split.x_train,
    )

    x_test = imputer.transform(
        split.x_test,
    )

    print("\n" + "=" * 70)
    print("Missing Value Imputation")
    print("=" * 70)

    print(
        f"X Train Missing Values : "
        f"{x_train.isna().sum().sum()}"
    )

    print(
        f"X Test Missing Values  : "
        f"{x_test.isna().sum().sum()}"
    )


    # ==========================================================
    # PHASE 9 - Categorical Encoding
    # ==========================================================

    encoder = CategoricalEncoder()

    # IMPORTANT:
    # Fit ONLY on training data.
    # Unknown categories in test data are handled
    # by the encoder.

    encoder.fit(
        x_train,
    )

    x_train = encoder.transform(
        x_train,
    )

    x_test = encoder.transform(
        x_test,
    )

    print("\n" + "=" * 70)
    print("Categorical Encoding")
    print("=" * 70)

    print("Data Types:")
    print(
        x_train.dtypes.value_counts()
    )

    print()

    object_columns = (
        x_train
        .select_dtypes(
            include=["object"]
        )
        .columns
    )

    print(
        f"Object Columns : "
        f"{len(object_columns)}"
    )


    # ==========================================================
    # PHASE 10 - Training
    # ==========================================================

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    trainer = ModelTrainer(
        model=model,
    )

    training_result = trainer.train(
        x_train,
        split.y_train,
    )

    print("\n" + "=" * 70)
    print("Baseline Model")
    print("=" * 70)

    print(
        f"Training Time : "
        f"{training_result.training_time:.2f} seconds"
    )


    # ==========================================================
    # PHASE 11 - Evaluation
    # ==========================================================

    evaluator = ModelEvaluator()

    evaluation_result = evaluator.evaluate(
        model=training_result.model,
        x_test=x_test,
        y_test=split.y_test,
    )

    print("\n" + "=" * 70)
    print("Evaluation")
    print("=" * 70)

    print(
        f"Accuracy  : "
        f"{evaluation_result.accuracy:.4f}"
    )

    print(
        f"Precision : "
        f"{evaluation_result.precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{evaluation_result.recall:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{evaluation_result.f1_score:.4f}"
    )

    print(
        f"ROC AUC   : "
        f"{evaluation_result.roc_auc:.4f}"
    )

    print("\nConfusion Matrix")

    print(
        evaluation_result.confusion_matrix
    )


    # ==========================================================
    # PHASE 12 - Feature Importance
    # ==========================================================

    importance_analyzer = (
        FeatureImportanceAnalyzer()
    )

    feature_importances = (
        importance_analyzer.analyze(
            model=training_result.model,
            feature_names=x_train.columns.tolist(),
        )
    )

    reporter = FeatureImportanceReporter()

    reporter.report(
        importances=feature_importances,
        top_n=20,
    )


if __name__ == "__main__":
    main()
