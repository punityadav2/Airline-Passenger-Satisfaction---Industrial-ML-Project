"""
Data preprocessing module for handling data cleaning, imputation, and transformations.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from src.utils.logger import logger
from src.config import (
    COLUMNS_TO_DROP,
    IMPUTATION_METHOD,
    KNN_NEIGHBORS,
    OUTLIER_METHOD,
    LOW_QUANTILE,
    UP_QUANTILE,
    Z_SCORE_THRESHOLD
)


class DataPreprocessor:
    """Handle data cleaning and preprocessing."""
    
    @staticmethod
    def drop_unnecessary_columns(
        df: pd.DataFrame,
        columns_to_drop: List[str] = None
    ) -> pd.DataFrame:
        """
        Drop unnecessary columns from the dataset.
        
        Args:
            df: Input DataFrame
            columns_to_drop: List of column names to drop
            
        Returns:
            DataFrame with dropped columns
        """
        if columns_to_drop is None:
            columns_to_drop = COLUMNS_TO_DROP
        
        cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        if cols_to_drop:
            logger.info(f"Dropping columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        return df
    
    @staticmethod
    def handle_missing_values(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        num_cols: List[str],
        method: str = None,
        n_neighbors: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handle missing values using specified imputation method.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            num_cols: List of numerical column names
            method: Imputation method ('knn', 'mean', 'median')
            n_neighbors: Number of neighbors for KNN imputation
            
        Returns:
            Tuple of (imputed_train_df, imputed_test_df)
        """
        if method is None:
            method = IMPUTATION_METHOD
        if n_neighbors is None:
            n_neighbors = KNN_NEIGHBORS
        
        train_df_copy = train_df.copy()
        test_df_copy = test_df.copy()
        
        if method == 'knn':
            logger.info(f"Applying KNN imputation with {n_neighbors} neighbors")
            imputer = KNNImputer(n_neighbors=n_neighbors)
            
            train_df_copy[num_cols] = imputer.fit_transform(train_df_copy[num_cols])
            test_df_copy[num_cols] = imputer.transform(test_df_copy[num_cols])
            
        elif method == 'mean':
            logger.info("Applying mean imputation")
            for col in num_cols:
                mean_val = train_df_copy[col].mean()
                train_df_copy[col].fillna(mean_val, inplace=True)
                test_df_copy[col].fillna(mean_val, inplace=True)
                
        elif method == 'median':
            logger.info("Applying median imputation")
            for col in num_cols:
                median_val = train_df_copy[col].median()
                train_df_copy[col].fillna(median_val, inplace=True)
                test_df_copy[col].fillna(median_val, inplace=True)
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        logger.info("Missing values handled successfully")
        return train_df_copy, test_df_copy
    
    @staticmethod
    def detect_outliers(
        df: pd.DataFrame,
        column: str,
        method: str = None,
        low_quantile: float = None,
        up_quantile: float = None
    ) -> Tuple[float, float]:
        """
        Calculate outlier thresholds for a column.
        
        Args:
            df: Input DataFrame
            column: Column name
            method: Detection method ('iqr' or 'z_score')
            low_quantile: Lower quantile
            up_quantile: Upper quantile
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if method is None:
            method = OUTLIER_METHOD
        if low_quantile is None:
            low_quantile = LOW_QUANTILE
        if up_quantile is None:
            up_quantile = UP_QUANTILE
        
        if method == 'iqr':
            q1 = df[column].quantile(low_quantile)
            q3 = df[column].quantile(up_quantile)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
        elif method == 'z_score':
            mean = df[column].mean()
            std_dev = df[column].std()
            lower_bound = mean - Z_SCORE_THRESHOLD * std_dev
            upper_bound = mean + Z_SCORE_THRESHOLD * std_dev
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return lower_bound, upper_bound
    
    @staticmethod
    def cap_outliers(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        num_cols: List[str],
        method: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Cap outliers at specified bounds.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            num_cols: List of numerical column names
            method: Outlier detection method
            
        Returns:
            Tuple of (treated_train_df, treated_test_df)
        """
        train_df_copy = train_df.copy()
        test_df_copy = test_df.copy()
        
        logger.info(f"Capping outliers using {method or OUTLIER_METHOD} method")
        
        for col in num_cols:
            lower_bound, upper_bound = DataPreprocessor.detect_outliers(train_df, col, method)
            
            # Apply bounds
            train_df_copy[col] = train_df_copy[col].clip(lower_bound, upper_bound)
            test_df_copy[col] = test_df_copy[col].clip(lower_bound, upper_bound)
        
        logger.info("Outliers capped successfully")
        return train_df_copy, test_df_copy
