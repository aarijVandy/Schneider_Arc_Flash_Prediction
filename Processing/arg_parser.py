"""
Argument Parser for Arc Flash Prediction Model
This module handles command-line argument parsing for training, testing, and prediction.
"""

import argparse
import os
from typing import Any


class ArgumentParser:
    """Class responsible for parsing command-line arguments"""
    
    def __init__(self):
        """Initialize ArgumentParser"""
        self.parser = argparse.ArgumentParser(
            description='Arc Flash Event Prediction System',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Train a model
  python main.py --mode train --data ./Data/arc_flash_data.csv --model_path ./Models/model.pkl
  
  # Test a model
  python main.py --mode test --data ./Data/test_data.csv --model_path ./Models/model.pkl
  
  # Make predictions
  python main.py --mode predict --data ./Data/new_data.csv --model_path ./Models/model.pkl --output ./Plots/predictions.csv
            """
        )
        self._add_arguments()
        
    def _add_arguments(self):
        """Add all command-line arguments"""
        
        # Mode argument (required)
        self.parser.add_argument(
            '--mode',
            type=str,
            required=True,
            choices=['train', 'test', 'predict'],
            help='Operation mode: train, test, or predict'
        )
        
        # Data path argument (required)
        self.parser.add_argument(
            '--data',
            type=str,
            required=True,
            help='Path to the dataset file (CSV, Excel, or JSON)'
        )
        
        # Model path argument
        self.parser.add_argument(
            '--model_path',
            type=str,
            default='./Models/arc_flash_model.pkl',
            help='Path to save/load the model (default: ./Models/arc_flash_model.pkl)'
        )
        
        # Target column name
        self.parser.add_argument(
            '--target',
            type=str,
            default='arcing_event',
            help='Name of the target column in the dataset (default: arcing_event)'
        )
        
        # Test size for train/validation split
        self.parser.add_argument(
            '--test_size',
            type=float,
            default=0.2,
            help='Proportion of data to use for validation during training (default: 0.2)'
        )
        
        # Random state for reproducibility
        self.parser.add_argument(
            '--random_state',
            type=int,
            default=42,
            help='Random seed for reproducibility (default: 42)'
        )
        
        # Data cleaning options
        self.parser.add_argument(
            '--missing_strategy',
            type=str,
            default='mean',
            choices=['mean', 'median', 'drop'],
            help='Strategy for handling missing values (default: mean)'
        )
        
        self.parser.add_argument(
            '--remove_duplicates',
            action='store_true',
            default=True,
            help='Remove duplicate rows from the dataset'
        )
        
        self.parser.add_argument(
            '--handle_outliers',
            action='store_true',
            default=False,
            help='Apply outlier detection and removal'
        )
        
        self.parser.add_argument(
            '--outlier_method',
            type=str,
            default='iqr',
            choices=['iqr', 'zscore'],
            help='Method for outlier detection (default: iqr)'
        )
        
        self.parser.add_argument(
            '--outlier_threshold',
            type=float,
            default=1.5,
            help='Threshold for outlier detection (default: 1.5 for IQR)'
        )
        
        # Model parameters
        self.parser.add_argument(
            '--model_type',
            type=str,
            default='random_forest',
            choices=['random_forest'],
            help='Type of machine learning model (default: random_forest)'
        )
        
        self.parser.add_argument(
            '--n_estimators',
            type=int,
            default=100,
            help='Number of trees in Random Forest (default: 100)'
        )
        
        self.parser.add_argument(
            '--max_depth',
            type=int,
            default=10,
            help='Maximum depth of trees in Random Forest (default: 10)'
        )
        
        # Output options
        self.parser.add_argument(
            '--output',
            type=str,
            default=None,
            help='Path to save predictions (for predict mode)'
        )
        
        self.parser.add_argument(
            '--verbose',
            action='store_true',
            default=False,
            help='Enable verbose output'
        )
        
        # Scaler and encoder paths (for saving/loading preprocessing objects)
        self.parser.add_argument(
            '--scaler_path',
            type=str,
            default='./Models/scaler.pkl',
            help='Path to save/load the scaler (default: ./Models/scaler.pkl)'
        )
        
        self.parser.add_argument(
            '--encoder_path',
            type=str,
            default='./Models/encoders.pkl',
            help='Path to save/load the label encoders (default: ./Models/encoders.pkl)'
        )
    
    def parse_args(self) -> argparse.Namespace:
        """
        Parse command-line arguments
        
        Returns:
            Parsed arguments as Namespace object
        """
        args = self.parser.parse_args()
        
        # Validate arguments
        self._validate_args(args)
        
        return args
    
    def _validate_args(self, args: argparse.Namespace):
        """
        Validate parsed arguments
        
        Args:
            args: Parsed arguments
        """
        # Check if data file exists
        if not os.path.exists(args.data):
            raise FileNotFoundError(f"Data file not found: {args.data}")
        
        # For test and predict modes, check if model exists
        if args.mode in ['test', 'predict']:
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(
                    f"Model file not found: {args.model_path}. "
                    f"Please train a model first or provide a valid model path."
                )
        
        # Validate test_size
        if not 0 < args.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {args.test_size}")
        
        # Validate model parameters
        if args.n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive, got {args.n_estimators}")
        
        if args.max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {args.max_depth}")
    
    def print_args(self, args: argparse.Namespace):
        """
        Print parsed arguments
        
        Args:
            args: Parsed arguments
        """
        print("\n" + "="*60)
        print("Arc Flash Prediction - Configuration")
        print("="*60)
        print(f"Mode: {args.mode}")
        print(f"Data Path: {args.data}")
        print(f"Model Path: {args.model_path}")
        print(f"Target Column: {args.target}")
        
        if args.mode == 'train':
            print(f"\nTraining Configuration:")
            print(f"  - Test Size: {args.test_size}")
            print(f"  - Random State: {args.random_state}")
            print(f"  - Model Type: {args.model_type}")
            print(f"  - N Estimators: {args.n_estimators}")
            print(f"  - Max Depth: {args.max_depth}")
            print(f"\nData Cleaning:")
            print(f"  - Missing Value Strategy: {args.missing_strategy}")
            print(f"  - Remove Duplicates: {args.remove_duplicates}")
            print(f"  - Handle Outliers: {args.handle_outliers}")
            if args.handle_outliers:
                print(f"  - Outlier Method: {args.outlier_method}")
                print(f"  - Outlier Threshold: {args.outlier_threshold}")
        
        if args.mode == 'predict' and args.output:
            print(f"\nOutput Path: {args.output}")
        
        print("="*60 + "\n")


def get_parser() -> ArgumentParser:
    """
    Factory function to get ArgumentParser instance
    
    Returns:
        ArgumentParser instance
    """
    return ArgumentParser()
