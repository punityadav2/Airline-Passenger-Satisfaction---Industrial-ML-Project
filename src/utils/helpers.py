"""
Helper functions and utilities for the Airline Passenger Satisfaction project.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Union
from src.utils.logger import logger


def grab_col_names(
    dataframe: pd.DataFrame,
    target_var: str,
    cat_th: int = 10,
    car_th: int = 20,
    print_results: bool = True
) -> Tuple[List[str], List[str], List[str], str]:
    """
    Identify and categorize columns into categorical and numerical.
    
    Args:
        dataframe: Input DataFrame
        target_var: Target variable name
        cat_th: Categorical threshold (unique values)
        car_th: Cardinal threshold (unique values)
        print_results: Whether to print results
        
    Returns:
        Tuple of (categorical_cols, numerical_cols, cardinal_cols, target_var)
    """
    cat_cols = [
        col for col in dataframe.columns
        if dataframe[col].dtype in ["category", "object", "bool"] and col != target_var
    ]

    num_but_cat = [
        col for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtype in ["int", "float"] 
        and col != target_var
    ]

    cat_but_car = [
        col for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtype in ["category", "object"] 
        and col != target_var
    ]

    cat_cols = list(set(cat_cols) - set(cat_but_car)) + num_but_cat
    num_cols = [
        col for col in dataframe.columns
        if dataframe[col].dtype in ["int", "float"] and col != target_var
    ]
    num_cols = list(set(num_cols) - set(cat_cols))

    if print_results:
        logger.info(f"Dataset Analysis:")
        logger.info(f"  Observations: {dataframe.shape[0]}")
        logger.info(f"  Variables: {dataframe.shape[1]}")
        logger.info(f"  Categorical Variables: {len(cat_cols)}")
        logger.info(f"  Numerical Variables: {len(num_cols)}")
        logger.info(f"  Cardinal Categorical Variables: {len(cat_but_car)}")
        logger.info(f"  Target Variable: {target_var}")

    return cat_cols, num_cols, cat_but_car, target_var


def missing_value_analysis(
    dataframe: pd.DataFrame,
    include_no_missing: bool = True
) -> pd.DataFrame:
    """
    Analyze and report missing values in the dataset.
    
    Args:
        dataframe: Input DataFrame
        include_no_missing: Whether to include columns with no missing values
        
    Returns:
        DataFrame with missing value statistics
    """
    missing_count = dataframe.isnull().sum()
    value_count = dataframe.shape[0]
    missing_percentage = round(missing_count / value_count * 100, 2)

    missing_df = pd.DataFrame({
        "count": missing_count,
        "percentage": missing_percentage
    })

    if not include_no_missing:
        missing_df = missing_df[missing_df['count'] > 0]

    missing_df = missing_df.sort_values(by='percentage', ascending=False)
    return missing_df


def get_unique_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Get unique value statistics for all columns.
    
    Args:
        dataframe: Input DataFrame
        
    Returns:
        DataFrame with unique value statistics
    """
    output_data = []

    for col in dataframe.columns:
        unique_count = dataframe[col].nunique()
        unique_values = dataframe[col].unique() if unique_count <= 10 else "-"
        output_data.append([col, unique_count, unique_values, dataframe[col].dtype])

    output_dataframe = pd.DataFrame(
        output_data,
        columns=['Column Name', 'Number of Unique Values', 'Unique Values (if ≤ 10)', 'Data Type']
    )
    return output_dataframe


def summarize_categorical_columns(df: pd.DataFrame, cat_cols: List[str]) -> None:
    """
    Print summary statistics for categorical columns.
    
    Args:
        df: Input DataFrame
        cat_cols: List of categorical column names
    """
    for col in cat_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

        value_counts = df[col].value_counts()
        ratio = 100 * value_counts / len(df)

        summary_df = pd.DataFrame({
            "Count": value_counts,
            "Ratio (%)": ratio
        })

        logger.info(f"\nDistribution of {col}:")
        logger.info(summary_df.to_string())


def examine_skewness(
    dataframe: pd.DataFrame,
    skew_threshold: float = 0.05,
    kurt_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Examine skewness and kurtosis of numerical columns.
    
    Args:
        dataframe: Input DataFrame with numerical columns
        skew_threshold: Threshold for skewness classification
        kurt_threshold: Threshold for kurtosis classification
        
    Returns:
        DataFrame with skewness and kurtosis analysis
    """
    skewness = dataframe.skew().sort_values(ascending=False)
    kurtosis = dataframe.kurtosis().sort_values(ascending=False)

    distribution_df = pd.DataFrame({
        'Skewness': skewness,
        'Kurtosis': kurtosis
    })

    skew_conditions = [
        abs(distribution_df['Skewness']) <= skew_threshold,
        distribution_df['Skewness'] > skew_threshold,
        distribution_df['Skewness'] < -skew_threshold
    ]
    skew_categories = ['Approximately Normal', 'Right-Skewed', 'Left-Skewed']

    kurt_conditions = [
        distribution_df['Kurtosis'] <= kurt_threshold,
        distribution_df['Kurtosis'] > kurt_threshold
    ]
    kurt_categories = ['Low Kurtosis (Flat)', 'High Kurtosis (Peaked)']

    distribution_df['Skew Type'] = np.select(skew_conditions, skew_categories, default='Unknown')
    distribution_df['Kurtosis Type'] = np.select(kurt_conditions, kurt_categories, default='Unknown')

    distribution_df = distribution_df.sort_values(by='Skewness', ascending=False)
    return distribution_df
