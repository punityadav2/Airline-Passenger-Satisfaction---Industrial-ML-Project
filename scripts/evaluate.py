"""
Evaluation script for the Airline Passenger Satisfaction project.
Evaluate the trained model on test data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data import DataLoader
from src.models import ModelTrainer, ModelEvaluator
from src.visualization import Plotter
from src.utils.logger import logger
from src.config import TARGET_COLUMN, FINAL_MODEL_PATH
from sklearn.metrics import confusion_matrix, roc_curve


def evaluate_model():
    """
    Evaluate the trained model.
    """
    logger.info("Starting model evaluation...")
    
    # Load model
    model = ModelTrainer.load_model(FINAL_MODEL_PATH)
    
    # Load processed data
    _, df_test = DataLoader.load_processed_data()
    
    # Prepare data
    X_test = df_test.drop(columns=TARGET_COLUMN, errors='ignore')
    y_test = df_test[TARGET_COLUMN]
    
    logger.info(f"Evaluating on {len(X_test)} test samples")
    
    # Get metrics
    metrics = ModelEvaluator.evaluate_model_test(model, X_test, y_test)
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("=" * 60)
    for metric, value in metrics.items():
        logger.info(f"{metric:.<40} {value:.4f}")
    
    # Get confusion matrix
    logger.info("\nComputing confusion matrix...")
    cm = ModelEvaluator.get_confusion_matrix(model, X_test, y_test)
    
    # Get classification report
    logger.info("\nClassification Report:")
    report = ModelEvaluator.get_classification_report(
        model, X_test, y_test,
        target_names=['Dissatisfied', 'Satisfied']
    )
    logger.info(report)
    
    # Get ROC curve
    logger.info("\nComputing ROC curve...")
    fpr, tpr, roc_auc = ModelEvaluator.get_roc_curve(model, X_test, y_test)
    
    # Visualizations
    logger.info("\nGenerating visualizations...")
    
    # Confusion matrix plot
    Plotter.plot_confusion_matrix(cm, labels=['Dissatisfied', 'Satisfied'])
    
    # ROC curve plot
    if roc_auc:
        Plotter.plot_roc_curve(fpr, tpr, roc_auc)
    
    # Feature importance
    feature_importance = ModelEvaluator.get_feature_importance(model, X_test.columns.tolist())
    if feature_importance is not None:
        Plotter.plot_feature_importance(
            feature_importance['Feature'].tolist(),
            feature_importance['Importance'].values
        )
    
    logger.info("\nEvaluation completed!")
    return metrics


if __name__ == "__main__":
    metrics = evaluate_model()
    print("\n✓ Model evaluation completed successfully!")
