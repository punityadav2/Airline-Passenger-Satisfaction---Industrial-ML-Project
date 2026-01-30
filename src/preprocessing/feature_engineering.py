"""
Feature engineering module for creating and transforming features.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from src.utils.logger import logger
from src.config import (
    AGE_BINS,
    AGE_LABELS,
    DISTANCE_BINS,
    DISTANCE_LABELS,
    DELAY_BINS,
    DELAY_LABELS,
    SCALING_METHOD,
    ENCODING_METHOD
)


class FeatureEngineer:
    """Handle feature creation and transformation."""
    
    @staticmethod
    def categorize_numeric_features(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create categorical bins from numerical features.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (transformed_train_df, transformed_test_df)
        """
        train_df_copy = train_df.copy()
        test_df_copy = test_df.copy()
        
        logger.info("Creating categorical features from numerical columns")
        
        for dataframe in [train_df_copy, test_df_copy]:
            # Age categorization
            if 'Age' in dataframe.columns:
                dataframe['age_cat'] = pd.cut(
                    dataframe['Age'],
                    bins=AGE_BINS,
                    labels=AGE_LABELS,
                    include_lowest=True
                )
            
            # Flight distance categorization
            if 'Flight Distance' in dataframe.columns:
                dataframe['flight_distance_cat'] = pd.cut(
                    dataframe['Flight Distance'],
                    bins=DISTANCE_BINS,
                    labels=DISTANCE_LABELS,
                    include_lowest=True
                )
            
            # Departure delay categorization
            if 'Departure Delay in Minutes' in dataframe.columns:
                dataframe['departure_delay_cat'] = pd.cut(
                    dataframe['Departure Delay in Minutes'],
                    bins=DELAY_BINS,
                    labels=DELAY_LABELS,
                    include_lowest=True
                )
        
        logger.info("Categorical features created successfully")
        return train_df_copy, test_df_copy
    
    @staticmethod
    def scale_features(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        num_cols: List[str],
        method: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
        """
        Scale numerical features.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            num_cols: List of numerical column names
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            Tuple of (scaled_train_df, scaled_test_df, scaler_object)
        """
        if method is None:
            method = SCALING_METHOD
        
        train_df_copy = train_df.copy()
        test_df_copy = test_df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        logger.info(f"Scaling features using {method} scaler")
        
        if num_cols:
            train_df_copy[num_cols] = scaler.fit_transform(train_df_copy[num_cols])
            test_df_copy[num_cols] = scaler.transform(test_df_copy[num_cols])
        
        logger.info("Features scaled successfully")
        return train_df_copy, test_df_copy, scaler
    
    @staticmethod
    def encode_categorical_features(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cat_cols: List[str],
        method: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Encode categorical features.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            cat_cols: List of categorical column names
            method: Encoding method ('label' or 'onehot')
            
        Returns:
            Tuple of (encoded_train_df, encoded_test_df, encoders_dict)
        """
        if method is None:
            method = ENCODING_METHOD
        
        train_df_copy = train_df.copy()
        test_df_copy = test_df.copy()
        encoders = {}
        
        logger.info(f"Encoding categorical features using {method} encoding")
        
        if method == 'label':
            for col in cat_cols:
                if col in train_df_copy.columns:
                    le = LabelEncoder()
                    train_df_copy[col] = le.fit_transform(train_df_copy[col].astype(str))
                    test_df_copy[col] = le.transform(test_df_copy[col].astype(str))
                    encoders[col] = le
                    
        elif method == 'onehot':
            train_df_copy = pd.get_dummies(train_df_copy, columns=cat_cols, drop_first=True)
            test_df_copy = pd.get_dummies(test_df_copy, columns=cat_cols, drop_first=True)
            
            # Ensure same columns in test as train
            missing_cols = set(train_df_copy.columns) - set(test_df_copy.columns)
            for col in missing_cols:
                test_df_copy[col] = 0
            
            test_df_copy = test_df_copy[train_df_copy.columns]
        
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        logger.info("Categorical features encoded successfully")
        return train_df_copy, test_df_copy, encoders
    
    @staticmethod
    def encode_target_variable(
        df: pd.DataFrame,
        target_col: str,
        method: str = 'label'
    ) -> Tuple[pd.DataFrame, object]:
        """
        Encode target variable.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            method: Encoding method ('label' or 'onehot')
            
        Returns:
            Tuple of (encoded_df, encoder_object)
        """
        df_copy = df.copy()
        
        if target_col not in df_copy.columns:
            return df_copy, None
        
        logger.info(f"Encoding target variable: {target_col}")
        
        if df_copy[target_col].dtype == 'object':
            if method == 'label':
                le = LabelEncoder()
                df_copy[target_col] = le.fit_transform(df_copy[target_col])
                logger.info("Target variable encoded successfully")
                return df_copy, le
                
            elif method == 'onehot':
                target_encoded = pd.get_dummies(df_copy[target_col], drop_first=True)
                df_copy = df_copy.drop(target_col, axis=1).join(target_encoded)
                logger.info("Target variable encoded successfully")
                return df_copy, None
        
        return df_copy, None
