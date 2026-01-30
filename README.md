# Airline Passenger Satisfaction - Industrial ML Project

A professional, modular machine learning project for predicting airline passenger satisfaction using structured data analysis, preprocessing, and model evaluation.

## 📋 Project Overview

This project analyzes and predicts passenger satisfaction levels using advanced data analysis and machine learning techniques. The codebase follows industry best practices with a modular, production-ready architecture.

> A production-grade machine learning system that predicts airline passenger satisfaction with **96.35% accuracy** using advanced gradient boosting algorithms.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](.)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Why This Approach?](#-why-this-approach)
- [Results & Improvements](#-results--improvements)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Key Features](#-key-features)
- [Documentation](#-documentation)

---

## 🎯 Problem Statement

### What problem did we solve?

Airlines struggle to understand and predict passenger satisfaction, leading to:

- **Lost Revenue**: Dissatisfied passengers don't return (customer churn)
- **Poor Resource Allocation**: Not knowing which services to improve
- **Reactive Approach**: Addressing issues only after complaints
- **Missed Opportunities**: Unable to identify at-risk passengers proactively

**Business Impact**: A 5% increase in customer retention can boost profits by 25-95% (Harvard Business Review)

### The Challenge

With thousands of flights daily and millions of passengers, airlines need to:
1. **Predict satisfaction** before passengers complete their journey
2. **Identify key drivers** of satisfaction and dissatisfaction
3. **Take proactive action** to improve customer experience
4. **Measure impact** of service improvements

---

## 💡 Solution Overview

### How did we solve it?

We built an **end-to-end machine learning pipeline** that:

#### 1. **Data Collection & Analysis**
- Analyzed **129,880 passenger records** from Kaggle
- 25 features including demographics, flight details, and 14 service ratings
- Comprehensive exploratory data analysis to understand patterns

#### 2. **Intelligent Preprocessing**
- **Missing Value Handling**: KNN imputation for accurate data completion
- **Outlier Treatment**: IQR-based capping to handle extreme values
- **Feature Engineering**: Created categorical bins and scaled features
- **Smart Encoding**: Label encoding for categorical variables

#### 3. **Advanced Model Development**
- **Baseline Comparison**: Evaluated 10 different algorithms
- **Hyperparameter Tuning**: Used Optuna (200 trials) for optimal parameters
- **Feature Selection**: SHAP analysis to identify 20 most important features
- **Final Model**: LightGBM with optimized hyperparameters

#### 4. **Production-Ready Implementation**
- Modular, scalable architecture following industry best practices
- Comprehensive logging and error handling
- Automated pipeline from raw data to predictions
- Easy deployment with saved model artifacts

### The Pipeline

```
Raw Data → Validation → Preprocessing → Feature Engineering → Model Training → Predictions
   ↓           ↓              ↓                 ↓                    ↓              ↓
129K rows   Clean data   Handle NaN      Scale & Encode      LightGBM Model   96.35% Accuracy
```

---

## 🔬 Why This Approach?

### Method Selection Rationale

#### **1. Why LightGBM over other algorithms?**

| Criteria | LightGBM | CatBoost | XGBoost | Random Forest |
|----------|----------|----------|---------|---------------|
| **Accuracy** | 96.38% ✅ | 96.41% | 96.26% | 96.10% |
| **Training Speed** | 4.5s ✅ | 75.97s | 2.61s | 10.33s |
| **ROC-AUC** | 99.48% ✅ | 99.51% | 99.47% | 99.33% |
| **Memory Usage** | Low ✅ | Medium | Medium | High |
| **Interpretability** | High ✅ | High | High | Medium |

**Decision**: LightGBM offers the **best balance** of accuracy, speed, and efficiency.

#### **2. Why SHAP for Feature Selection?**

- **Interpretability**: Explains individual predictions
- **Consistency**: Based on game theory (Shapley values)
- **Actionable**: Identifies which features to improve
- **Reduced Complexity**: 20 features instead of 24 (16.7% reduction)

#### **3. Why KNN Imputation over Mean/Median?**

- **Context-Aware**: Uses similar passengers' data
- **Better Accuracy**: Preserves relationships between features
- **Realistic Values**: Imputed values are more representative

#### **4. Why This Architecture?**

```python
src/
├── config/          # Centralized configuration
├── data/            # Data loading & validation
├── preprocessing/   # Reusable preprocessing modules
├── models/          # Model training & evaluation
├── visualization/   # Plotting utilities
└── utils/           # Logging & helpers
```

**Benefits**:
- ✅ **Modularity**: Easy to modify individual components
- ✅ **Reusability**: Functions can be used in notebooks or scripts
- ✅ **Maintainability**: Clear separation of concerns
- ✅ **Scalability**: Easy to add new features or models
- ✅ **Production-Ready**: Follows industry best practices

---

## 📊 Results & Improvements

### What improved?

#### **Model Performance**

| Metric | Baseline (Logistic Regression) | Our Model (LightGBM) | Improvement |
|--------|-------------------------------|----------------------|-------------|
| **Accuracy** | 87.45% | **96.35%** | **+10.2%** 🚀 |
| **Precision** | 87.44% | **97.02%** | **+11.0%** 🚀 |
| **Recall** | 87.45% | **94.60%** | **+8.2%** 🚀 |
| **ROC-AUC** | 92.64% | **99.50%** | **+7.4%** 🚀 |

#### **Business Impact**

**Before (Manual Analysis)**:
- ❌ Reactive approach to customer complaints
- ❌ No prediction capability
- ❌ Unknown satisfaction drivers
- ❌ Generic service improvements

**After (ML-Powered)**:
- ✅ **96.35% accuracy** in predicting satisfaction
- ✅ **Proactive intervention** for at-risk passengers
- ✅ **Data-driven insights** on what to improve
- ✅ **Targeted improvements** based on feature importance

#### **Key Discoveries**

Our model revealed the **Top 5 Satisfaction Drivers**:

1. 🛫 **Flight Distance** (Importance: 11,759)
   - Long-haul flights need more attention
   - Premium services more critical for longer flights

2. 👤 **Passenger Age** (Importance: 9,158)
   - Different age groups have different needs
   - Tailor services based on demographics

3. ⏰ **Departure Delays** (Importance: 4,113)
   - Major dissatisfaction factor
   - Operational efficiency is critical

4. 🪑 **Leg Room Service** (Importance: 2,827)
   - Comfort is a top priority
   - Consider seat redesign or premium options

5. ✅ **Check-in Service** (Importance: 2,755)
   - First impression matters
   - Streamline check-in processes

#### **Prediction Accuracy Breakdown**

```
Confusion Matrix (25,976 test samples):
                    Predicted
                 Dissatisfied  Satisfied
Actual
Dissatisfied        14,242        331      ← 97.73% correct
Satisfied              616     10,787      ← 94.60% correct
```

**Error Analysis**:
- Only **3.65% error rate** (947 misclassifications out of 25,976)
- **False Positive Rate**: 2.27% (low risk of over-promising)
- **False Negative Rate**: 5.40% (acceptable for proactive interventions)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 8 GB RAM (minimum), 16 GB recommended
- 500 MB free disk space

### Installation

#### Step 1: Navigate to Project Directory
```bash
cd "Analysis of Airline Passenger Satisfaction"
```

#### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

#### Step 3: Activate Virtual Environment

**Windows (PowerShell)**:
```bash
.\venv\Scripts\Activate.ps1
```

**Windows (CMD)**:
```bash
venv\Scripts\activate.bat
```

**macOS/Linux**:
```bash
source venv/bin/activate
```

#### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- pandas, numpy (data processing)
- scikit-learn (ML algorithms)
- lightgbm, catboost, xgboost (gradient boosting)
- optuna (hyperparameter tuning)
- shap (model explainability)
- matplotlib, seaborn, plotly (visualization)

---

### Running the Project

#### **Option 1: Complete Pipeline (Recommended)**

Run the entire workflow from start to finish:

```bash
python main.py
```

**What it does**:
1. ✅ Loads and validates raw data
2. ✅ Preprocesses data (handles missing values, outliers)
3. ✅ Engineers features (scaling, encoding)
4. ✅ Trains baseline models (10 algorithms)
5. ✅ Tunes hyperparameters (Optuna optimization)
6. ✅ Selects features (SHAP analysis)
7. ✅ Trains final model (LightGBM)
8. ✅ Evaluates performance
9. ✅ Saves model to `artifacts/final_model.pkl`

**Expected time**: ~35 minutes (first run)

---

#### **Option 2: Individual Scripts**

Run specific parts of the pipeline:

**Train the model**:
```bash
python scripts/train.py
```

**Make predictions on test data**:
```bash
python scripts/predict.py
```
- Outputs: `artifacts/predictions.csv`
- Includes confidence scores for each prediction

**Evaluate model performance**:
```bash
python scripts/evaluate.py
```
- Generates confusion matrix
- Plots ROC curve
- Shows feature importance
- Saves evaluation report

---

#### **Option 3: Interactive Notebooks**

Explore the analysis interactively:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

**Available Notebooks**:
- `01_eda.ipynb` - Exploratory Data Analysis with visualizations
- `02_modeling.ipynb` - Model training and evaluation experiments

---

### Verify Installation

Test that everything is working:

```bash
# Test imports
python -c "from src.data import DataLoader; print('✓ DataLoader works')"
python -c "from src.models import ModelTrainer; print('✓ ModelTrainer works')"

# Check if model exists
python -c "import os; print('✓ Model exists' if os.path.exists('artifacts/final_model.pkl') else '✗ Run main.py first')"
```

---

## 📁 Project Structure

```
Analysis of Airline Passenger Satisfaction/
│
├── 📂 src/                          # Core source code
│   ├── 📂 config/
│   │   └── config.py                # Centralized configuration & parameters
│   │
│   ├── 📂 data/
│   │   ├── data_loader.py           # Load & save data files
│   │   └── data_validator.py        # Data quality checks
│   │
│   ├── 📂 preprocessing/
│   │   ├── preprocessor.py          # Data cleaning & imputation
│   │   └── feature_engineering.py   # Feature transformation & encoding
│   │
│   ├── 📂 models/
│   │   ├── model_trainer.py         # Model training & tuning
│   │   └── model_evaluator.py       # Model evaluation & metrics
│   │
│   ├── 📂 visualization/
│   │   └── plotter.py               # Visualization functions
│   │
│   └── 📂 utils/
│       ├── logger.py                # Logging setup
│       └── helpers.py               # Utility functions
│
├── 📂 scripts/                      # Executable scripts
│   ├── train.py                     # Training workflow
│   ├── predict.py                   # Prediction script
│   └── evaluate.py                  # Evaluation script
│
├── 📂 notebooks/                    # Jupyter notebooks
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   └── 02_modeling.ipynb           # Model Training & Evaluation
│
├── 📂 data/
│   ├── 📂 raw/                      # Original dataset
│   │   ├── train.csv               # 103,904 training samples
│   │   └── test.csv                # 25,976 test samples
│   └── 📂 processed/                # Preprocessed data
│       ├── train_preprocessed.csv
│       └── test_preprocessed.csv
│
├── 📂 artifacts/                    # Generated outputs
│   ├── final_model.pkl             # Trained LightGBM model (6.2 MB)
│   └── predictions.csv             # Test set predictions (2.4 MB)
│
├── 📂 logs/                         # Execution logs
│   └── airline_satisfaction.log    # Application logs
│
├── 📄 main.py                       # Main pipeline orchestrator
├── 📄 requirements.txt              # Project dependencies
├── 📄 README.md                     # This file
├── 📄 PROJECT_REPORT.md             # Comprehensive project report
├── 📄 IMPLEMENTATION_PLAN.md        # Detailed architecture guide
├── 📄 VERSION_UPGRADE.md            # Dependency upgrade notes
└── 📄 QUICK_REFERENCE.py           # Code examples & patterns
```

---

## 🎯 Key Features

### ✨ What Makes This Project Special?

#### **1. Production-Ready Architecture**
- ✅ Modular design with clear separation of concerns
- ✅ Centralized configuration management
- ✅ Comprehensive error handling and logging
- ✅ Easy to maintain and extend

#### **2. Multiple Model Support**
- ✅ 10 baseline algorithms evaluated
- ✅ LightGBM, CatBoost, XGBoost with hyperparameter tuning
- ✅ Easy to add new models

#### **3. Advanced Feature Engineering**
- ✅ KNN imputation for missing values
- ✅ IQR-based outlier capping
- ✅ Multiple scaling methods (Standard, MinMax, Robust)
- ✅ Flexible encoding (Label, One-Hot)

#### **4. Model Explainability**
- ✅ SHAP-based feature importance
- ✅ Feature selection with interpretability
- ✅ Confusion matrix and classification reports
- ✅ ROC curves and performance metrics

#### **5. Comprehensive Evaluation**
- ✅ 5-fold cross-validation
- ✅ Multiple metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- ✅ Detailed classification reports
- ✅ Visual performance analysis

#### **6. Flexible Interfaces**
- ✅ Command-line scripts for automation
- ✅ Jupyter notebooks for exploration
- ✅ Python modules for integration
- ✅ Saved models for deployment

---

## 🔧 Customization

### Change Model Parameters

Edit `src/config/config.py`:

```python
# Hyperparameter tuning settings
OPTUNA_N_TRIALS = 200        # Number of optimization trials
CV_FOLDS = 5                 # Cross-validation folds

# LightGBM parameters
LIGHTGBM_PARAMS = {
    'learning_rate': 0.1633,
    'n_estimators': 484,
    'max_depth': 15,
    # ... more parameters
}
```

### Change Preprocessing Settings

```python
# Imputation method
IMPUTATION_METHOD = 'knn'    # Options: 'knn', 'mean', 'median'

# Scaling method
SCALING_METHOD = 'standard'   # Options: 'standard', 'minmax', 'robust'

# Encoding method
ENCODING_METHOD = 'label'     # Options: 'label', 'onehot'

# Outlier handling
OUTLIER_METHOD = 'iqr'       # Options: 'iqr', 'z_score'
```

---

## 📊 Model Performance Summary

### Final Model Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 96.35% | 96 out of 100 predictions are correct |
| **Precision** | 97.02% | When we predict "Satisfied", we're right 97% of the time |
| **Recall** | 94.60% | We correctly identify 95% of actually satisfied passengers |
| **F1-Score** | 95.80% | Balanced measure of precision and recall |
| **ROC-AUC** | 99.50% | Excellent discrimination between classes |

### Top 10 Feature Importance

1. **Flight Distance** - Most influential factor
2. **Age** - Demographics matter
3. **Departure Delay** - Timeliness is critical
4. **Leg Room Service** - Comfort is key
5. **Check-in Service** - First impressions count
6. **On-board Service** - Service quality matters
7. **Gate Location** - Convenience is important
8. **Seat Comfort** - Physical comfort drives satisfaction
9. **Inflight Service** - Attentive service pays off
10. **Departure/Arrival Time** - Scheduling convenience

---

## 📚 Documentation

### Available Documentation

- **[README.md](README.md)** (this file) - Quick start and overview
- **[PROJECT_REPORT.md](PROJECT_REPORT.md)** - Comprehensive project report
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Detailed architecture guide
- **[VERSION_UPGRADE.md](VERSION_UPGRADE.md)** - Dependency upgrade notes
- **[QUICK_REFERENCE.py](QUICK_REFERENCE.py)** - Code examples & usage patterns

### Code Documentation

All modules include comprehensive docstrings:

```python
from src.models import ModelTrainer

# View documentation
help(ModelTrainer.train_tuned_lightgbm)
```

---

## 🛠️ Troubleshooting

### Common Issues

**Issue**: Model file not found when running `predict.py`
```bash
# Solution: Train the model first
python main.py
# or
python scripts/train.py
```

**Issue**: Import errors
```bash
# Solution: Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # macOS/Linux

# Reinstall dependencies
pip install -r requirements.txt
```

**Issue**: Out of memory errors
```bash
# Solution: Reduce batch size or use smaller dataset
# Edit src/config/config.py and reduce OPTUNA_N_TRIALS
```

**Issue**: Slow training
```bash
# Solution: Reduce number of trials or use faster model
# In src/config/config.py:
OPTUNA_N_TRIALS = 50  # Instead of 200
```

---

## 📈 Future Enhancements

### Planned Improvements

- [ ] **REST API** - Flask/FastAPI endpoint for real-time predictions
- [ ] **Dashboard** - Interactive Streamlit/Dash dashboard
- [ ] **Docker** - Containerization for easy deployment
- [ ] **CI/CD** - Automated testing and deployment pipeline
- [ ] **Model Monitoring** - Track performance over time
- [ ] **A/B Testing** - Compare model versions
- [ ] **Multi-class** - Predict satisfaction levels (1-5 stars)
- [ ] **Time Series** - Incorporate temporal patterns

---

## 📊 Dataset Information

**Source**: [Kaggle - Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

**Size**:
- Training: 103,904 samples
- Test: 25,976 samples
- Total: 129,880 passenger records

**Features** (25 total):
- **Demographics**: Gender, Age, Customer Type
- **Flight Info**: Class, Type of Travel, Flight Distance
- **Service Ratings** (14 features, scale 1-5):
  - Inflight wifi service
  - Departure/Arrival time convenient
  - Ease of Online booking
  - Gate location
  - Food and drink
  - Online boarding
  - Seat comfort
  - Inflight entertainment
  - On-board service
  - Leg room service
  - Baggage handling
  - Check-in service
  - Inflight service
  - Cleanliness
- **Delays**: Departure Delay, Arrival Delay (minutes)
- **Target**: Satisfaction (Satisfied/Dissatisfied)

---

## 💻 System Requirements

**Minimum**:
- Python 3.8+
- 8 GB RAM
- 2 CPU cores
- 500 MB disk space

**Recommended**:
- Python 3.10+
- 16 GB RAM
- 4+ CPU cores
- 1 GB disk space

---

## 📝 License

This project is provided for educational purposes.

---

## 🤝 Contributing

Contributions are welcome! When extending this project:

1. Follow the existing module structure
2. Add comprehensive docstrings to functions
3. Use the centralized logger for logging
4. Update configuration in `src/config/config.py`
5. Add unit tests for new functionality

---

## 📞 Support

For issues or questions:

1. Check the logs: `logs/airline_satisfaction.log`
2. Review documentation: `PROJECT_REPORT.md`
3. Verify configuration: `src/config/config.py`
4. Check module docstrings for API documentation

---

## 🎓 Learning Resources

### Understanding the Code

- **Data Processing**: See `src/data/` modules
- **Preprocessing**: See `src/preprocessing/` modules
- **Model Training**: See `src/models/model_trainer.py`
- **Evaluation**: See `src/models/model_evaluator.py`

### Key Concepts Used

- **Gradient Boosting**: LightGBM, CatBoost, XGBoost
- **Hyperparameter Tuning**: Optuna (Bayesian optimization)
- **Feature Selection**: SHAP (Shapley values)
- **Cross-Validation**: Stratified K-Fold
- **Imputation**: KNN imputation
- **Scaling**: StandardScaler, MinMaxScaler, RobustScaler

---

## 📌 Quick Reference

### Essential Commands

```bash
# Activate environment
.\venv\Scripts\Activate.ps1

# Run complete pipeline
python main.py

# Train model only
python scripts/train.py

# Make predictions
python scripts/predict.py

# Evaluate model
python scripts/evaluate.py

# Launch Jupyter
jupyter notebook
```

### Key Files

- **Model**: `artifacts/final_model.pkl`
- **Predictions**: `artifacts/predictions.csv`
- **Logs**: `logs/airline_satisfaction.log`
- **Config**: `src/config/config.py`

---

**Last Updated**: January 30, 2026  
**Project Status**: ✅ Production-Ready  
**Model Version**: 1.0  
**Accuracy**: 96.35% | **ROC-AUC**: 99.50%

---

<div align="center">

**Built with ❤️ using Python, LightGBM, and SHAP**

[📊 View Report](PROJECT_REPORT.md) | [📖 Documentation](IMPLEMENTATION_PLAN.md) | [💻 Code Examples](QUICK_REFERENCE.py)

</div>
