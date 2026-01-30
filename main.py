"""
Main entry point for the Airline Passenger Satisfaction project.
Orchestrates the entire ML pipeline from data loading to model evaluation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.utils.logger import logger, setup_logger
from src.data import DataLoader, DataValidator
from src.preprocessing import DataPreprocessor, FeatureEngineer
from src.utils.helpers import grab_col_names
from src.models import ModelTrainer, ModelEvaluator
from src.visualization import Plotter
from src.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, SHAP_THRESHOLD

from sklearn.model_selection import train_test_split
import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Feature selection will be skipped.")


def run_eda_pipeline():
    """Run Exploratory Data Analysis pipeline."""
    logger.info("=" * 80)
    logger.info("STARTING EDA PIPELINE")
    logger.info("=" * 80)
    
    # Load raw data
    df_train, df_test = DataLoader.load_raw_data()
    
    # Validate data
    logger.info("\nValidating data...")
    DataValidator.check_missing_values(df_train)
    DataValidator.check_missing_values(df_test)
    DataValidator.check_duplicates(df_train)
    DataValidator.check_duplicates(df_test)
    DataValidator.validate_shapes(df_train, df_test)
    
    # Get column information
    cat_cols, num_cols, cat_but_car, target_col = grab_col_names(
        df_train, TARGET_COLUMN, print_results=True
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("EDA PIPELINE COMPLETED")
    logger.info("=" * 80)
    
    return df_train, df_test, cat_cols, num_cols


def run_preprocessing_pipeline(df_train, df_test, num_cols, cat_cols):
    """Run data preprocessing pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 80)
    
    # Drop unnecessary columns
    logger.info("\nDropping unnecessary columns...")
    df_train = DataPreprocessor.drop_unnecessary_columns(df_train)
    df_test = DataPreprocessor.drop_unnecessary_columns(df_test)
    
    # Re-grab columns after dropping
    cat_cols, num_cols, _, _ = grab_col_names(df_train, TARGET_COLUMN, print_results=False)
    
    # Handle missing values
    logger.info("\nHandling missing values...")
    df_train, df_test = DataPreprocessor.handle_missing_values(
        df_train, df_test, num_cols
    )
    
    # Cap outliers
    logger.info("\nCapping outliers...")
    df_train, df_test = DataPreprocessor.cap_outliers(df_train, df_test, num_cols)
    
    # Feature engineering
    logger.info("\nCreating new features...")
    df_train, df_test = FeatureEngineer.categorize_numeric_features(df_train, df_test)
    
    # Re-grab columns after feature engineering
    cat_cols, num_cols, _, _ = grab_col_names(df_train, TARGET_COLUMN, print_results=False)
    
    # Scale numerical features
    logger.info("\nScaling numerical features...")
    df_train, df_test, scaler = FeatureEngineer.scale_features(
        df_train, df_test, num_cols
    )
    
    # Encode categorical features
    logger.info("\nEncoding categorical features...")
    df_train, df_test, encoders = FeatureEngineer.encode_categorical_features(
        df_train, df_test, cat_cols
    )
    
    # Encode target variable
    logger.info("\nEncoding target variable...")
    df_train, le_target = FeatureEngineer.encode_target_variable(
        df_train, TARGET_COLUMN
    )
    df_test, _ = FeatureEngineer.encode_target_variable(
        df_test, TARGET_COLUMN
    )
    
    # Save processed data
    logger.info("\nSaving preprocessed data...")
    DataLoader.save_processed_data(df_train, df_test)
    
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING PIPELINE COMPLETED")
    logger.info("=" * 80)
    
    return df_train, df_test


def run_modeling_pipeline(df_train, df_test):
    """Run model training and evaluation pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("STARTING MODELING PIPELINE")
    logger.info("=" * 80)
    
    # Prepare data
    logger.info("\nPreparing training and test data...")
    X = df_train.drop(columns=TARGET_COLUMN)
    y = df_train[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Evaluate baseline models
    logger.info("\nEvaluating baseline models...")
    baseline_results = ModelTrainer.evaluate_baseline_models(X_train, y_train)
    logger.info(f"\nBaseline Results:\n{baseline_results.to_string()}")
    
    # Train tuned LightGBM
    logger.info("\nTraining tuned LightGBM model...")
    final_model, best_params = ModelTrainer.train_tuned_lightgbm(
        X_train, X_test, y_train, y_test
    )
    
    # Feature selection using SHAP (if available)
    if SHAP_AVAILABLE:
        logger.info("\nPerforming SHAP-based feature selection...")
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer(X_train)
        
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        selected_features = X_train.columns[feature_importance > SHAP_THRESHOLD].tolist()
        excluded_features = X_train.columns[feature_importance <= SHAP_THRESHOLD].tolist()
        
        logger.info(f"Selected {len(selected_features)} features out of {X_train.shape[1]}")
        logger.info(f"Selected features: {selected_features}")
        logger.info(f"Excluded features: {excluded_features}")
        
        # Train model with selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        logger.info("\nTraining final model with selected features...")
        final_model, _ = ModelTrainer.train_tuned_lightgbm(
            X_train_selected, X_test_selected, y_train, y_test
        )
        
        X_eval = X_test_selected
    else:
        X_eval = X_test
    
    # Evaluate final model
    logger.info("\nEvaluating final model on test set...")
    test_metrics = ModelEvaluator.evaluate_model_test(final_model, X_eval, y_test)
    logger.info(f"\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Get confusion matrix and classification report
    cm = ModelEvaluator.get_confusion_matrix(final_model, X_eval, y_test)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    report = ModelEvaluator.get_classification_report(final_model, X_eval, y_test)
    logger.info(f"\nClassification Report:\n{report}")
    
    # Get ROC curve
    fpr, tpr, roc_auc = ModelEvaluator.get_roc_curve(final_model, X_eval, y_test)
    if roc_auc:
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
    
    # Save model
    logger.info("\nSaving final model...")
    ModelTrainer.save_model(final_model)
    
    logger.info("\n" + "=" * 80)
    logger.info("MODELING PIPELINE COMPLETED")
    logger.info("=" * 80)
    
    return final_model, test_metrics


def main():
    """Run the complete ML pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("AIRLINE PASSENGER SATISFACTION - ML PIPELINE")
    logger.info("=" * 80)
    
    try:
        # EDA
        df_train, df_test, cat_cols, num_cols = run_eda_pipeline()
        
        # Preprocessing
        df_train_processed, df_test_processed = run_preprocessing_pipeline(
            df_train, df_test, num_cols, cat_cols
        )
        
        # Modeling
        final_model, metrics = run_modeling_pipeline(df_train_processed, df_test_processed)
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return final_model, metrics
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
