"""
Visualization module for the Airline Passenger Satisfaction project.
Contains modular plotting functions for EDA and model analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from src.utils.logger import logger
from src.config import (
    PALETTE_TRAIN,
    PALETTE_TEST,
    PALETTE_HEATMAP,
    FIGURE_SIZE_MEDIUM,
    FIGURE_SIZE_LARGE,
    DPI
)


class Plotter:
    """Modular plotting functions for data visualization."""
    
    @staticmethod
    def plot_histograms(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        num_cols: List[str],
        n_cols: int = 3
    ) -> None:
        """
        Plot histograms comparing train and test numerical distributions.
        
        Args:
            df_train: Training DataFrame
            df_test: Test DataFrame
            num_cols: List of numerical column names
            n_cols: Number of columns in subplot grid
        """
        logger.info("Plotting numerical feature distributions")
        
        n_features = len(num_cols)
        n_rows = int(np.ceil(n_features / n_cols))

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 4 * n_rows), dpi=DPI)
        axes = axes.flatten()

        for i, col in enumerate(num_cols):
            ax = axes[i]
            sns.histplot(df_train[col], kde=True, label='Train', ax=ax, color='skyblue')
            sns.histplot(df_test[col], kde=True, label='Test', ax=ax, color='coral')
            ax.set_title(f'{col} Distribution (Train vs Test)', fontsize=12, fontweight='bold')
            ax.legend()

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle('Comparison of Numerical Feature Distributions: Train vs Test',
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_categorical_distributions(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        cat_cols: List[str],
        n_cols: int = 2
    ) -> None:
        """
        Plot categorical feature distributions.
        
        Args:
            df_train: Training DataFrame
            df_test: Test DataFrame
            cat_cols: List of categorical column names
            n_cols: Number of columns in subplot grid
        """
        logger.info("Plotting categorical feature distributions")
        
        n_features = len(cat_cols) * 2  # 2 plots per feature (train & test)
        n_rows = int(np.ceil(n_features / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows), dpi=DPI)
        if n_rows == 1 or n_cols == 1:
            axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)
        axes = axes.flatten()

        def annotate_bars(ax):
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='bottom', fontsize=9)

        for i, col in enumerate(cat_cols):
            ax_train = axes[2 * i]
            sns.countplot(data=df_train, x=col, ax=ax_train, palette=PALETTE_TRAIN)
            ax_train.set_title(f'{col} (Train)', fontsize=12, fontweight='bold')
            ax_train.set_ylabel('Count')
            ax_train.set_xlabel('')
            annotate_bars(ax_train)
            ax_train.tick_params(axis='x', rotation=45)

            ax_test = axes[2 * i + 1]
            sns.countplot(data=df_test, x=col, ax=ax_test, palette=PALETTE_TEST)
            ax_test.set_title(f'{col} (Test)', fontsize=12, fontweight='bold')
            ax_test.set_ylabel('Count')
            ax_test.set_xlabel('')
            annotate_bars(ax_test)
            ax_test.tick_params(axis='x', rotation=45)

        # Hide extra subplots
        for j in range(2 * n_features, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle('Comparison of Categorical Feature Distributions: Train vs Test',
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_heatmap(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        num_cols: List[str]
    ) -> None:
        """
        Plot correlation heatmaps for train and test data.
        
        Args:
            df_train: Training DataFrame
            df_test: Test DataFrame
            num_cols: List of numerical column names
        """
        logger.info("Plotting correlation heatmaps")
        
        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LARGE, dpi=DPI)

        dfs = [df_train[num_cols], df_test[num_cols]]
        titles = ['Train Dataset Correlation', 'Test Dataset Correlation']

        for ax, df, title in zip(axes, dfs, titles):
            corr_matrix = df.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap=PALETTE_HEATMAP, annot=True,
                       fmt=".2f", cbar=True, annot_kws={"size": 10}, linewidths=0.5,
                       linecolor='white', square=True, ax=ax, vmin=-1, vmax=1, center=0)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', labelsize=10, rotation=45)
            ax.tick_params(axis='y', labelsize=10)

        fig.suptitle('Correlation Matrices: Train vs Test Datasets',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_target_distribution(
        df: pd.DataFrame,
        target_col: str
    ) -> None:
        """
        Plot target variable distribution.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
        """
        logger.info(f"Plotting target variable distribution: {target_col}")
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
        
        value_counts = df[target_col].value_counts()
        colors = ['#FF6B6B', '#4ECDC4']
        value_counts.plot(kind='bar', ax=ax, color=colors[:len(value_counts)])
        
        ax.set_title(f'Distribution of {target_col}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        ax.set_xlabel(target_col)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        labels: List[str] = None,
        title: str = 'Confusion Matrix'
    ) -> None:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            cm: Confusion matrix array
            labels: Class labels
            title: Plot title
        """
        logger.info(f"Plotting confusion matrix")
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                   xticklabels=labels or ['Negative', 'Positive'],
                   yticklabels=labels or ['Negative', 'Positive'])
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_roc_curve(
        fpr: np.ndarray,
        tpr: np.ndarray,
        roc_auc: float
    ) -> None:
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rate
            tpr: True positive rate
            roc_auc: AUC score
        """
        logger.info("Plotting ROC curve")
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
        
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})',
               color='darkorange', lw=2)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_importance(
        features: List[str],
        importances: np.ndarray,
        top_n: int = 20
    ) -> None:
        """
        Plot feature importances.
        
        Args:
            features: Feature names
            importances: Feature importance values
            top_n: Number of top features to display
        """
        logger.info(f"Plotting top {top_n} feature importances")
        
        # Create DataFrame for sorting
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)
        
        sns.barplot(data=importance_df, y='Feature', x='Importance', ax=ax, palette='viridis')
        
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_learning_curve(
        train_scores: np.ndarray,
        test_scores: np.ndarray,
        metric_name: str = 'Accuracy'
    ) -> None:
        """
        Plot learning curve.
        
        Args:
            train_scores: Training set scores
            test_scores: Test set scores
            metric_name: Name of the metric
        """
        logger.info("Plotting learning curve")
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
        
        ax.plot(range(1, len(train_scores) + 1), train_scores, 'o-', label='Train',
               linewidth=2, markersize=8, color='blue')
        ax.plot(range(1, len(test_scores) + 1), test_scores, 's-', label='Test',
               linewidth=2, markersize=8, color='red')
        
        ax.set_xlabel('Fold Number', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'Learning Curve - {metric_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
