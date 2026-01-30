"""
Configuration management for the Airline Passenger Satisfaction project.
Centralized configuration for paths, hyperparameters, and other constants.
"""

from pathlib import Path
import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Raw data file paths
TRAIN_RAW_PATH = RAW_DATA_DIR / "train.csv"
TEST_RAW_PATH = RAW_DATA_DIR / "test.csv"

# Processed data file paths
TRAIN_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_preprocessed.csv"
TEST_PROCESSED_PATH = PROCESSED_DATA_DIR / "test_preprocessed.csv"

# Model artifacts
FINAL_MODEL_PATH = ARTIFACTS_DIR / "final_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
ENCODERS_PATH = ARTIFACTS_DIR / "encoders.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.pkl"

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
TARGET_COLUMN = "satisfaction"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Column categorization thresholds
CATEGORICAL_THRESHOLD = 10
CARDINAL_THRESHOLD = 20

# ============================================================================
# DATA PREPROCESSING CONFIGURATION
# ============================================================================
# Missing value imputation
IMPUTATION_METHOD = "knn"  # Options: 'knn', 'mean', 'median'
KNN_NEIGHBORS = 5

# Outlier detection and treatment
OUTLIER_METHOD = "iqr"  # Options: 'iqr', 'z_score'
OUTLIER_HANDLING = "cap"  # Options: 'cap', 'remove'
LOW_QUANTILE = 0.10
UP_QUANTILE = 0.90
Z_SCORE_THRESHOLD = 3

# Feature engineering
AGE_BINS = [0, 12, 19, 35, 50, 65, 100]
AGE_LABELS = ["Child", "Teenager", "Young Adult", "Adult", "Middle Aged", "Senior"]

DISTANCE_BINS = [0, 1500, 3500, float("inf")]
DISTANCE_LABELS = ["Short Haul", "Medium Haul", "Long Haul"]

DELAY_BINS = [-float("inf"), 0, 30, 60, 120, float("inf")]
DELAY_LABELS = ["On Time", "Slightly Delayed", "Moderately Delayed", "Heavily Delayed", "Severely Delayed"]

# Scaling and encoding
SCALING_METHOD = "standard"  # Options: 'standard', 'minmax', 'robust'
ENCODING_METHOD = "label"    # Options: 'label', 'onehot'

# Columns to drop during cleaning
COLUMNS_TO_DROP = ["Unnamed: 0", "id", "Arrival Delay in Minutes"]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Cross-validation
CV_FOLDS = 5

# Baseline models to evaluate
BASELINE_MODELS = [
    "LogisticRegression",
    "KNN",
    "DecisionTree",
    "RandomForest",
    "GradientBoosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "NaiveBayes",
    "BalancedRandomForest"
]

# ============================================================================
# HYPERPARAMETER TUNING CONFIGURATION
# ============================================================================
# Optuna
OPTUNA_N_TRIALS = 200
OPTUNA_N_WARMUP_STEPS = 10

# LightGBM hyperparameters (best from tuning)
LIGHTGBM_PARAMS = {
    "learning_rate": 0.1633,
    "n_estimators": 484,
    "max_depth": 15,
    "subsample": 0.8724,
    "colsample_bytree": 0.7263,
    "reg_lambda": 3.41e-04,
    "reg_alpha": 1.26e-01,
    "num_leaves": 121,
    "min_child_samples": 57,
}

# CatBoost hyperparameters (best from tuning)
CATBOOST_PARAMS = {
    "iterations": 900,
    "learning_rate": 0.0325,
    "depth": 8,
    "l2_leaf_reg": 1.5,
    "border_count": 200,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 5.0,
    "eval_metric": "Accuracy",
    "loss_function": "Logloss",
    "verbose": False,
}

# XGBoost hyperparameters (best from tuning)
XGBOOST_PARAMS = {
    "booster": "gbtree",
    "learning_rate": 0.1,
    "n_estimators": 300,
    "max_depth": 7,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_lambda": 0.5,
    "reg_alpha": 0.1,
    "gamma": 0.2,
    "min_child_weight": 5,
    "grow_policy": "depthwise",
    "use_label_encoder": False,
    "eval_metric": "auc",
}

# ============================================================================
# FEATURE SELECTION
# ============================================================================
SHAP_THRESHOLD = 0.1

# ============================================================================
# EVALUATION METRICS
# ============================================================================
SCORING_METRICS = {
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "f1": "f1_weighted",
    "roc_auc": "roc_auc"
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / "airline_satisfaction.log"

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_SIZE_SMALL = (12, 6)
FIGURE_SIZE_MEDIUM = (16, 8)
FIGURE_SIZE_LARGE = (20, 12)
DPI = 100

# Color palettes
PALETTE_TRAIN = "GnBu_r"
PALETTE_TEST = "OrRd_r"
PALETTE_HEATMAP = "RdBu_r"
