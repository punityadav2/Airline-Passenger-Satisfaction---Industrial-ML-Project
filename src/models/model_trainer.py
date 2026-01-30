"""
Model training module for the Airline Passenger Satisfaction project.
Handles baseline model evaluation, hyperparameter tuning, and model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import time
import joblib
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.utils.logger import logger
from src.config import (
    CV_FOLDS,
    SCORING_METRICS,
    LIGHTGBM_PARAMS,
    CATBOOST_PARAMS,
    XGBOOST_PARAMS,
    FINAL_MODEL_PATH
)


class ModelTrainer:
    """Handle model training and evaluation."""
    
    @staticmethod
    def get_baseline_models() -> Dict[str, object]:
        """
        Get dictionary of baseline models.
        
        Returns:
            Dictionary of model name and model object
        """
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
            'XGBClassifier': XGBClassifier(eval_metric='logloss', random_state=42),
            'LGBMClassifier': LGBMClassifier(random_state=42, verbose=-1),
            'CatBoostClassifier': CatBoostClassifier(verbose=False, random_state=42),
            'GaussianNB': GaussianNB(),
            'BalancedRandomForestClassifier': BalancedRandomForestClassifier(random_state=42)
        }
        return models
    
    @staticmethod
    def evaluate_baseline_models(
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = None
    ) -> pd.DataFrame:
        """
        Evaluate baseline models using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv: Number of CV folds
            
        Returns:
            DataFrame with baseline model evaluation results
        """
        if cv is None:
            cv = CV_FOLDS
        
        logger.info("Evaluating baseline models...")
        
        models = ModelTrainer.get_baseline_models()
        
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'roc_auc': 'roc_auc'
        }
        
        results = []
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            try:
                start_time = time.time()
                cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring,
                                           n_jobs=-1, return_train_score=False)
                fit_time = time.time() - start_time
                
                result = {
                    'Model': name,
                    'Accuracy': round(cv_results['test_accuracy'].mean(), 4),
                    'Precision': round(cv_results['test_precision'].mean(), 4),
                    'Recall': round(cv_results['test_recall'].mean(), 4),
                    'F1-Score': round(cv_results['test_f1'].mean(), 4),
                    'ROC-AUC': round(cv_results['test_roc_auc'].mean(), 4),
                    'Fit-Time': round(fit_time, 2)
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        logger.info("Baseline model evaluation completed")
        return results_df
    
    @staticmethod
    def train_tuned_lightgbm(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        params: Dict = None
    ) -> Tuple[LGBMClassifier, Dict]:
        """
        Train LightGBM with optimized hyperparameters.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            params: Hyperparameters (uses defaults if None)
            
        Returns:
            Tuple of (trained_model, model_params)
        """
        if params is None:
            params = LIGHTGBM_PARAMS.copy()
        
        logger.info(f"Training LightGBM with params: {params}")
        
        model = LGBMClassifier(**params, verbose=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc'
        )
        
        logger.info("LightGBM training completed")
        return model, params
    
    @staticmethod
    def train_tuned_catboost(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        params: Dict = None
    ) -> Tuple[CatBoostClassifier, Dict]:
        """
        Train CatBoost with optimized hyperparameters.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            params: Hyperparameters (uses defaults if None)
            
        Returns:
            Tuple of (trained_model, model_params)
        """
        if params is None:
            params = CATBOOST_PARAMS.copy()
        
        logger.info(f"Training CatBoost with params: {params}")
        
        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=100,
            verbose=False
        )
        
        logger.info("CatBoost training completed")
        return model, params
    
    @staticmethod
    def train_tuned_xgboost(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        params: Dict = None
    ) -> Tuple[XGBClassifier, Dict]:
        """
        Train XGBoost with optimized hyperparameters.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            params: Hyperparameters (uses defaults if None)
            
        Returns:
            Tuple of (trained_model, model_params)
        """
        if params is None:
            params = XGBOOST_PARAMS.copy()
        
        logger.info(f"Training XGBoost with params: {params}")
        
        model = XGBClassifier(**params, random_state=42)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        logger.info("XGBoost training completed")
        return model, params
    
    @staticmethod
    def save_model(model: object, save_path: str = None) -> None:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model object
            save_path: Path to save model (uses default if None)
        """
        if save_path is None:
            save_path = FINAL_MODEL_PATH
        
        logger.info(f"Saving model to {save_path}")
        joblib.dump(model, save_path)
        logger.info("Model saved successfully")
    
    @staticmethod
    def load_model(save_path: str = None) -> object:
        """
        Load trained model from disk.
        
        Args:
            save_path: Path to load model (uses default if None)
            
        Returns:
            Loaded model object
        """
        if save_path is None:
            save_path = FINAL_MODEL_PATH
        
        logger.info(f"Loading model from {save_path}")
        model = joblib.load(save_path)
        logger.info("Model loaded successfully")
        return model
