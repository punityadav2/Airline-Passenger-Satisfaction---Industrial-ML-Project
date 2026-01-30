"""
Data loading and handling module for the Airline Passenger Satisfaction project.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
from src.utils.logger import logger
from src.config import TRAIN_RAW_PATH, TEST_RAW_PATH, TRAIN_PROCESSED_PATH, TEST_PROCESSED_PATH


class DataLoader:
    """Handle loading of raw and processed data."""
    
    @staticmethod
    def load_raw_data(train_path: Path = None, test_path: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw training and test data.
        
        Args:
            train_path: Path to raw training data
            test_path: Path to raw test data
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if train_path is None:
            train_path = TRAIN_RAW_PATH
        if test_path is None:
            test_path = TEST_RAW_PATH
            
        try:
            logger.info(f"Loading raw training data from {train_path}")
            df_train = pd.read_csv(train_path)
            
            logger.info(f"Loading raw test data from {test_path}")
            df_test = pd.read_csv(test_path)
            
            logger.info(f"Training data shape: {df_train.shape}")
            logger.info(f"Test data shape: {df_test.shape}")
            
            return df_train, df_test
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    @staticmethod
    def load_processed_data(
        train_path: Path = None,
        test_path: Path = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed training and test data.
        
        Args:
            train_path: Path to preprocessed training data
            test_path: Path to preprocessed test data
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if train_path is None:
            train_path = TRAIN_PROCESSED_PATH
        if test_path is None:
            test_path = TEST_PROCESSED_PATH
            
        try:
            logger.info(f"Loading processed training data from {train_path}")
            df_train = pd.read_csv(train_path)
            
            logger.info(f"Loading processed test data from {test_path}")
            df_test = pd.read_csv(test_path)
            
            logger.info(f"Processed training data shape: {df_train.shape}")
            logger.info(f"Processed test data shape: {df_test.shape}")
            
            return df_train, df_test
            
        except FileNotFoundError as e:
            logger.error(f"Processed data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    @staticmethod
    def save_processed_data(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_path: Path = None,
        test_path: Path = None
    ) -> None:
        """
        Save preprocessed data to CSV files.
        
        Args:
            train_df: Preprocessed training DataFrame
            test_df: Preprocessed test DataFrame
            train_path: Path to save training data
            test_path: Path to save test data
        """
        if train_path is None:
            train_path = TRAIN_PROCESSED_PATH
        if test_path is None:
            test_path = TEST_PROCESSED_PATH
            
        try:
            logger.info(f"Saving processed training data to {train_path}")
            train_df.to_csv(train_path, index=False)
            
            logger.info(f"Saving processed test data to {test_path}")
            test_df.to_csv(test_path, index=False)
            
            logger.info("Processed data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
