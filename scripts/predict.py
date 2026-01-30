"""
Prediction script for the Airline Passenger Satisfaction project.
Load a trained model and make predictions on new data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.models import ModelTrainer
from src.utils.logger import logger
from src.config import FINAL_MODEL_PATH
from src.data import DataLoader


def predict(input_data):
    """
    Make predictions on new data.
    
    Args:
        input_data: DataFrame with features for prediction
        
    Returns:
        predictions: Array of predictions
        probabilities: Array of prediction probabilities (if available)
    """
    try:
        logger.info("Loading trained model for prediction...")
        model = ModelTrainer.load_model(FINAL_MODEL_PATH)
        logger.info(f"✓ Model loaded successfully from {FINAL_MODEL_PATH}")
        
        logger.info(f"Making predictions on {len(input_data)} samples...")
        predictions = model.predict(input_data)
        
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_data)
            logger.info(f"✓ Predictions completed with confidence scores")
        else:
            logger.info(f"✓ Predictions completed")
        
        return predictions, probabilities
    
    except FileNotFoundError:
        logger.error(f"Model file not found at {FINAL_MODEL_PATH}")
        logger.error("Please run 'python main.py' or 'python scripts/train.py' first to train a model")
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise


def display_predictions(predictions, probabilities, input_data):
    """
    Display predictions in a readable format.
    
    Args:
        predictions: Array of predictions
        probabilities: Array of probabilities (optional)
        input_data: Original input DataFrame
    """
    results_df = input_data.copy()
    results_df['prediction'] = predictions
    
    if probabilities is not None:
        # Add probability for each class
        for i, prob in enumerate(probabilities[:5]):  # Show first 5 rows
            logger.info(f"Sample {i}: Prediction={predictions[i]}, Confidence={max(prob):.2%}")
    
    logger.info(f"\n📊 Prediction Summary:")
    logger.info(f"Total samples: {len(predictions)}")
    logger.info(f"Unique predictions: {set(predictions)}")
    
    # Count predictions
    unique, counts = np.unique(predictions, return_counts=True)
    for pred, count in zip(unique, counts):
        percentage = (count / len(predictions)) * 100
        logger.info(f"  Class '{pred}': {count} samples ({percentage:.1f}%)")
    
    return results_df


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PREDICTION SCRIPT - Airline Passenger Satisfaction")
    logger.info("=" * 60)
    
    try:
        # Load processed test data
        logger.info("\nLoading processed test data...")
        _, df_test = DataLoader.load_processed_data()
        logger.info(f"✓ Loaded {len(df_test)} test samples with {df_test.shape[1]} features")
        
        # Features selected during training (via SHAP feature selection)
        SELECTED_FEATURES = [
            'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
            'Inflight wifi service', 'Departure/Arrival time convenient', 
            'Ease of Online booking', 'Gate location', 'Food and drink',
            'Online boarding', 'Seat comfort', 'Inflight entertainment',
            'On-board service', 'Leg room service', 'Baggage handling',
            'Checkin service', 'Inflight service', 'Cleanliness', 
            'Departure Delay in Minutes'
        ]
        
        # Select only the features used during training
        logger.info(f"Selecting {len(SELECTED_FEATURES)} features for prediction...")
        X_test = df_test[SELECTED_FEATURES]
        logger.info(f"✓ Features selected: {X_test.shape[1]} features")
        
        # Make predictions
        logger.info("\n" + "-" * 60)
        predictions, probabilities = predict(X_test)
        logger.info("-" * 60)
        
        # Display results
        logger.info("\n")
        results_df = display_predictions(predictions, probabilities, X_test)
        
        # Save predictions
        output_path = Path(__file__).parent.parent / "artifacts" / "predictions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Predictions saved to {output_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ Prediction completed successfully!")
        logger.info("=" * 60)
    
    except Exception as e:
        logger.error(f"\n✗ Prediction failed: {str(e)}")
        sys.exit(1)
