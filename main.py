"""
Main execution script for Arc Flash Prediction System
This script orchestrates the training, testing, and prediction workflows.
"""

import sys
import os
import pandas as pd
import joblib
import logging
from pathlib import Path

# Add Processing directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arc_flash_model import (
    DataLoader, DataCleaner, Model, Train, Test, Forward
)
from arg_parser import get_parser

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArcFlashPipeline:
    """Main pipeline class to orchestrate the arc flash prediction workflow"""
    
    def __init__(self, args):
        """
        Initialize the pipeline with parsed arguments
        
        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.data_loader = None
        self.data_cleaner = None
        self.model = None
        self.trainer = None
        self.tester = None
        self.predictor = None
        
    def load_and_clean_data(self) -> tuple:
        """
        Load and clean the dataset
        
        Returns:
            Tuple of (X, y) - features and target
        """
        logger.info("Loading data...")
        self.data_loader = DataLoader(self.args.data)
        data = self.data_loader.load_data()
        
        logger.info("Cleaning data...")
        self.data_cleaner = DataCleaner(data, target_column=self.args.target)
        
        # Apply cleaning steps
        self.data_cleaner.handle_missing_values(strategy=self.args.missing_strategy)
        
        if self.args.remove_duplicates:
            self.data_cleaner.remove_duplicates()
        
        if self.args.handle_outliers:
            self.data_cleaner.handle_outliers(
                method=self.args.outlier_method,
                threshold=self.args.outlier_threshold
            )
        
        self.data_cleaner.encode_categorical_features()
        self.data_cleaner.normalize_features()
        
        # Split features and target
        X, y = self.data_cleaner.split_features_target()
        logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def train_mode(self):
        """Execute training workflow"""
        logger.info("="*60)
        logger.info("TRAINING MODE")
        logger.info("="*60)
        
        # Load and clean data
        X, y = self.load_and_clean_data()
        
        # Initialize model
        logger.info("Initializing model...")
        model_params = {
            'n_estimators': self.args.n_estimators,
            'max_depth': self.args.max_depth,
            'random_state': self.args.random_state
        }
        self.model = Model(model_type=self.args.model_type, **model_params)
        
        # Train model
        logger.info("Training model...")
        self.trainer = Train(
            self.model,
            test_size=self.args.test_size,
            random_state=self.args.random_state
        )
        metrics = self.trainer.train_model(X, y)
        
        # Save model
        logger.info("Saving model and preprocessing objects...")
        self.model.save_model(self.args.model_path)
        
        # Save scaler and encoders
        joblib.dump(self.data_cleaner.scaler, self.args.scaler_path)
        joblib.dump(self.data_cleaner.label_encoders, self.args.encoder_path)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {self.args.model_path}")
        logger.info(f"Scaler saved to: {self.args.scaler_path}")
        logger.info(f"Encoders saved to: {self.args.encoder_path}")
        
        return metrics
    
    def test_mode(self):
        """Execute testing workflow"""
        logger.info("="*60)
        logger.info("TESTING MODE")
        logger.info("="*60)
        
        # Load preprocessing objects
        logger.info("Loading preprocessing objects...")
        if not os.path.exists(self.args.scaler_path):
            logger.warning(f"Scaler not found at {self.args.scaler_path}. Using fresh scaler.")
        if not os.path.exists(self.args.encoder_path):
            logger.warning(f"Encoders not found at {self.args.encoder_path}. Using fresh encoders.")
        
        # Load and clean data
        X, y = self.load_and_clean_data()
        
        # Load model
        logger.info("Loading trained model...")
        self.model = Model(model_type=self.args.model_type)
        self.model.load_model(self.args.model_path)
        
        # Test model
        logger.info("Testing model...")
        self.tester = Test(self.model)
        metrics = self.tester.evaluate(X, y)
        
        logger.info("Testing completed successfully!")
        
        return metrics
    
    def predict_mode(self):
        """Execute prediction workflow"""
        logger.info("="*60)
        logger.info("PREDICTION MODE")
        logger.info("="*60)
        
        # Load and clean data (without target)
        logger.info("Loading data for prediction...")
        self.data_loader = DataLoader(self.args.data)
        data = self.data_loader.load_data()
        
        # Check if target column exists (if not, we're predicting)
        has_target = self.args.target in data.columns
        
        if has_target:
            logger.info(f"Target column '{self.args.target}' found. Will be excluded from features.")
        
        # Clean data
        logger.info("Preprocessing data...")
        self.data_cleaner = DataCleaner(data, target_column=self.args.target)
        
        # Apply same cleaning steps as training
        self.data_cleaner.handle_missing_values(strategy=self.args.missing_strategy)
        
        if self.args.remove_duplicates:
            self.data_cleaner.remove_duplicates()
        
        self.data_cleaner.encode_categorical_features()
        self.data_cleaner.normalize_features()
        
        if has_target:
            X, _ = self.data_cleaner.split_features_target()
        else:
            X = self.data_cleaner.get_cleaned_data()
        
        # Load model
        logger.info("Loading trained model...")
        self.model = Model(model_type=self.args.model_type)
        self.model.load_model(self.args.model_path)
        
        # Make predictions
        logger.info("Making predictions...")
        self.predictor = Forward(self.model, self.data_cleaner)
        predictions = self.predictor.predict(X)
        probabilities = self.predictor.predict_proba(X)
        
        # Create results dataframe
        results = pd.DataFrame({
            'prediction': predictions,
            'probability_class_0': probabilities[:, 0],
            'probability_class_1': probabilities[:, 1] if probabilities.shape[1] > 1 else 0
        })
        
        # Add original data
        results = pd.concat([data.reset_index(drop=True), results], axis=1)
        
        # Save predictions if output path specified
        if self.args.output:
            output_dir = os.path.dirname(self.args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            results.to_csv(self.args.output, index=False)
            logger.info(f"Predictions saved to: {self.args.output}")
        else:
            logger.info("\nPrediction Results (first 10 rows):")
            logger.info(f"\n{results.head(10)}")
        
        logger.info("Prediction completed successfully!")
        
        return results
    
    def run(self):
        """Execute the appropriate workflow based on mode"""
        try:
            if self.args.mode == 'train':
                return self.train_mode()
            elif self.args.mode == 'test':
                return self.test_mode()
            elif self.args.mode == 'predict':
                return self.predict_mode()
            else:
                raise ValueError(f"Invalid mode: {self.args.mode}")
                
        except Exception as e:
            logger.error(f"Error during execution: {str(e)}")
            raise


def main():
    """Main entry point"""
    try:
        # Parse arguments
        parser = get_parser()
        args = parser.parse_args()
        
        # Print configuration
        if args.verbose:
            parser.print_args(args)
        
        # Create and run pipeline
        pipeline = ArcFlashPipeline(args)
        pipeline.run()
        
        logger.info("\n" + "="*60)
        logger.info("EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("="*60 + "\n")
        
    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nExecution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
