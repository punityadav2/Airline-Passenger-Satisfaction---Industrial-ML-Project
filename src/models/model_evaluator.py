"""
Model evaluation module for assessing model performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import time

from src.utils.logger import logger
from src.config import CV_FOLDS, SCORING_METRICS


class ModelEvaluator:
    """Evaluate model performance."""
    
    @staticmethod
    def evaluate_model_cv(
        model: object,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = None
    ) -> Dict:
        """
        Evaluate model using cross-validation.
        
        Args:
            model: Trained model object
            X: Feature matrix
            y: Target vector
            cv: Number of CV folds
            
        Returns:
            Dictionary with CV evaluation results
        """
        if cv is None:
            cv = CV_FOLDS
        
        logger.info(f"Evaluating model using {cv}-fold cross-validation")
        
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=SCORING_METRICS,
            n_jobs=-1
        )
        
        results = {
            'Accuracy': {
                'Mean': np.mean(cv_results['test_accuracy']),
                'Std': np.std(cv_results['test_accuracy'])
            },
            'Precision': {
                'Mean': np.mean(cv_results['test_precision']),
                'Std': np.std(cv_results['test_precision'])
            },
            'Recall': {
                'Mean': np.mean(cv_results['test_recall']),
                'Std': np.std(cv_results['test_recall'])
            },
            'F1-Score': {
                'Mean': np.mean(cv_results['test_f1']),
                'Std': np.std(cv_results['test_f1'])
            },
            'ROC-AUC': {
                'Mean': np.mean(cv_results['test_roc_auc']),
                'Std': np.std(cv_results['test_roc_auc'])
            }
        }
        
        logger.info(f"Cross-validation evaluation completed")
        return results
    
    @staticmethod
    def evaluate_model_test(
        model: object,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model object
            X_test: Test feature matrix
            y_test: Test target vector
            
        Returns:
            Dictionary with test set evaluation metrics
        """
        logger.info("Evaluating model on test set")
        
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            y_pred_proba = None
            roc_auc = None
        
        metrics = {
            'Accuracy': float(accuracy_score(y_test, y_pred)),
            'Precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'Recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'F1-Score': float(f1_score(y_test, y_pred, zero_division=0)),
        }
        
        if roc_auc is not None:
            metrics['ROC-AUC'] = float(roc_auc)
        
        logger.info(f"Test set metrics: {metrics}")
        return metrics
    
    @staticmethod
    def get_confusion_matrix(
        model: object,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> np.ndarray:
        """
        Get confusion matrix for the model.
        
        Args:
            model: Trained model object
            X_test: Test feature matrix
            y_test: Test target vector
            
        Returns:
            Confusion matrix
        """
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        return cm
    
    @staticmethod
    def get_classification_report(
        model: object,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_names: list = None
    ) -> str:
        """
        Get classification report.
        
        Args:
            model: Trained model object
            X_test: Test feature matrix
            y_test: Test target vector
            target_names: Names of target classes
            
        Returns:
            Classification report string
        """
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
        logger.info(f"Classification Report:\n{report}")
        return report
    
    @staticmethod
    def get_roc_curve(
        model: object,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get ROC curve components.
        
        Args:
            model: Trained model object
            X_test: Test feature matrix
            y_test: Test target vector
            
        Returns:
            Tuple of (fpr, tpr, roc_auc)
        """
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model does not support probability prediction")
            return None, None, None
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        return fpr, tpr, roc_auc
    
    @staticmethod
    def compare_model_features(
        model: object,
        X_full: pd.DataFrame,
        X_selected: pd.DataFrame,
        y: pd.Series,
        cv: int = None
    ) -> pd.DataFrame:
        """
        Compare model performance with full vs selected features.
        
        Args:
            model: Model to evaluate
            X_full: Full feature set
            X_selected: Selected feature set
            y: Target vector
            cv: Number of CV folds
            
        Returns:
            DataFrame comparing performance
        """
        logger.info("Comparing full features vs selected features")
        
        if cv is None:
            cv = CV_FOLDS
        
        # Evaluate with full features
        start = time.time()
        full_results = ModelEvaluator.evaluate_model_cv(model, X_full, y, cv)
        full_time = time.time() - start
        
        # Evaluate with selected features
        start = time.time()
        selected_results = ModelEvaluator.evaluate_model_cv(model, X_selected, y, cv)
        selected_time = time.time() - start
        
        # Create comparison DataFrame
        comparison_data = {}
        for metric in full_results.keys():
            comparison_data[f'{metric} (Full)'] = [
                round(full_results[metric]['Mean'], 4),
                round(full_results[metric]['Std'], 4)
            ]
            comparison_data[f'{metric} (Selected)'] = [
                round(selected_results[metric]['Mean'], 4),
                round(selected_results[metric]['Std'], 4)
            ]
        
        comparison_df = pd.DataFrame(comparison_data, index=['Mean', 'Std']).T
        
        logger.info(f"Full features training time: {full_time:.2f}s")
        logger.info(f"Selected features training time: {selected_time:.2f}s")
        
        return comparison_df
    
    @staticmethod
    def get_feature_importance(
        model: object,
        feature_names: list = None
    ) -> pd.DataFrame:
        """
        Get feature importances from the model.
        
        Args:
            model: Trained model object
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        logger.info(f"Top 10 features:\n{importance_df.head(10).to_string()}")
        return importance_df
