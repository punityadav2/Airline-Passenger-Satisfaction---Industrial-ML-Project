"""
Data validation module for the Airline Passenger Satisfaction project.
"""

import pandas as pd
from typing import List
from src.utils.logger import logger


class DataValidator:
    """Validate data quality and integrity."""
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> dict:
        """
        Check for missing values in the dataset.
        
        Args:
            df: Input DataFrame
            threshold: Missing value percentage threshold (0-1)
            
        Returns:
            Dictionary with missing value statistics
        """
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        issues = {}
        for col in df.columns:
            if missing_pct[col] > threshold * 100:
                issues[col] = float(missing_pct[col])
        
        if issues:
            logger.warning(f"Found columns with missing values > {threshold*100}%: {issues}")
        
        return {
            "missing_count": missing_count.to_dict(),
            "missing_percentage": missing_pct.to_dict(),
            "issues": issues
        }
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame) -> int:
        """
        Check for duplicate rows.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Number of duplicate rows
        """
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate rows")
        else:
            logger.info("No duplicate rows found")
        
        return duplicate_count
    
    @staticmethod
    def check_data_types(df: pd.DataFrame, expected_types: dict = None) -> dict:
        """
        Check data types of columns.
        
        Args:
            df: Input DataFrame
            expected_types: Dictionary of expected column types
            
        Returns:
            Dictionary with data type information
        """
        current_types = df.dtypes.to_dict()
        
        issues = {}
        if expected_types:
            for col, expected_type in expected_types.items():
                if col in df.columns and str(current_types[col]) != expected_type:
                    issues[col] = {
                        "expected": expected_type,
                        "actual": str(current_types[col])
                    }
        
        if issues:
            logger.warning(f"Found data type mismatches: {issues}")
        
        return {
            "current_types": {k: str(v) for k, v in current_types.items()},
            "issues": issues
        }
    
    @staticmethod
    def check_value_ranges(df: pd.DataFrame, num_cols: List[str]) -> dict:
        """
        Check for unusual value ranges in numerical columns.
        
        Args:
            df: Input DataFrame
            num_cols: List of numerical column names
            
        Returns:
            Dictionary with value range statistics
        """
        issues = {}
        stats = {}
        
        for col in num_cols:
            if col in df.columns:
                stats[col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std())
                }
                
                # Check for negative values in typically non-negative columns
                if df[col].min() < 0 and "age" in col.lower() or "distance" in col.lower():
                    issues[col] = f"Found negative values in {col}"
        
        if issues:
            logger.warning(f"Found value range issues: {issues}")
        
        return {
            "statistics": stats,
            "issues": issues
        }
    
    @staticmethod
    def validate_shapes(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """
        Validate that train and test data have consistent column structure.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            True if validation passes
        """
        # Check if number of columns is similar (test might have target column)
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        
        missing_in_test = train_cols - test_cols
        extra_in_test = test_cols - train_cols
        
        if missing_in_test and missing_in_test != {"satisfaction"}:
            logger.warning(f"Columns in train but not in test: {missing_in_test}")
        if extra_in_test and extra_in_test != {"satisfaction"}:
            logger.warning(f"Columns in test but not in train: {extra_in_test}")
        
        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return True
