"""
Training script for the Airline Passenger Satisfaction project.
Run the complete ML pipeline from data loading to model evaluation.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import main

if __name__ == "__main__":
    print("\nStarting training pipeline...\n")
    model, metrics = main()
    print("\nTraining completed!")
    print(f"Final metrics: {metrics}")
