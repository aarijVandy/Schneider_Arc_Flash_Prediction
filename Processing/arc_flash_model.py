"""
Arc Flash Prediction Model
This module contains OOP classes for data loading, cleaning, model definition, 
training, testing, and forward prediction for arc flash events.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from typing import Tuple, Optional, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """Class responsible for loading dataset from various sources"""
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader with data path
        
        Args:
            data_path: Path to the dataset file
        """
        self.data_path = data_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from file (supports CSV, Excel, JSON)

        """
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found at: {self.data_path}")
            
            file_extension = os.path.splitext(self.data_path)[1].lower()
            
            if file_extension == '.csv':
                self.data = pd.read_csv(self.data_path)
                logger.info(f"Loaded CSV data with shape: {self.data.shape}")
            elif file_extension in ['.xlsx', '.xls']:
                self.data = pd.read_excel(self.data_path)
                logger.info(f"Loaded Excel data with shape: {self.data.shape}")
            elif file_extension == '.json':
                self.data = pd.read_json(self.data_path)
                logger.info(f"Loaded JSON data with shape: {self.data.shape}")
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            return self.data
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise


class DataCleaner:
    """Class responsible for cleaning and preprocessing data"""
    
    def __init__(self, data: pd.DataFrame, target_column: str = 'arcing_event'):
        """
        Initialize DataCleaner
        """
        self.data = data.copy()
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def handle_missing_values(self, strategy: str = 'mean') -> 'DataCleaner':
        """
        Handle missing values in the dataset
        Returns:
            Self for method chaining
        """
        logger.info(f"Missing values before cleaning: {self.data.isnull().sum().sum()}")
        
        if strategy == 'drop':
            self.data = self.data.dropna()
        elif strategy == 'mean':
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].median())
        
        logger.info(f"Missing values after cleaning: {self.data.isnull().sum().sum()}")
        return self
    
    
    def encode_categorical_features(self) -> 'DataCleaner':
        """
        Encode categorical features using Label Encoding
        
        Returns:
            Self for method chaining
        """
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != self.target_column:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded categorical column: {col}")
        
        return self
    
    def normalize_features(self, exclude_target: bool = True) -> 'DataCleaner':
        """
        Normalize numerical features using StandardScaler
        
        Args:
            exclude_target: Whether to exclude target column from normalization
        
        Returns:
            Self for method chaining
        """
        if exclude_target and self.target_column in self.data.columns:
            feature_cols = [col for col in self.data.columns if col != self.target_column]
        else:
            feature_cols = self.data.columns.tolist()
        
        self.feature_columns = feature_cols
        self.data[feature_cols] = self.scaler.fit_transform(self.data[feature_cols])
        logger.info(f"Normalized {len(feature_cols)} feature columns")
        
        return self
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the processed dataset
        """
        return self.data
    
    def split_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
    
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        return X, y


class Model:
    """Class responsible for model definition and management"""
    
    def __init__(self, model_type: str = 'random_forest', **model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the machine learning model"""
        if self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(self.model_params)
            self.model = RandomForestClassifier(**default_params)
            logger.info(f"Initialized Random Forest model with params: {default_params}")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def get_model(self):
        """Get the model instance"""
        return self.model
    
    def save_model(self, path: str):

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        
        self.model = joblib.load(path)
        logger.info(f"Model loaded from: {path}")


class Train:
    """Class responsible for model training"""
    
    def __init__(self, model: Model, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize Train
        
        Args:
            model: Model instance to train
            test_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Split data into training and validation sets
        """
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        logger.info(f"Training set size: {self.X_train.shape[0]}")
        logger.info(f"Validation set size: {self.X_val.shape[0]}")
    
    def train_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        logger.info("Starting model training...")
        self.prepare_data(X, y)
        
        self.model.get_model().fit(self.X_train, self.y_train)
        logger.info("Model training completed")
        
        # Validate on validation set
        train_score = self.model.get_model().score(self.X_train, self.y_train)
        val_score = self.model.get_model().score(self.X_val, self.y_val)
        
        logger.info(f"Training accuracy: {train_score:.4f}")
        logger.info(f"Validation accuracy: {val_score:.4f}")
        
        return {
            'train_accuracy': train_score,
            'val_accuracy': val_score
        }


class Test:
    """Class responsible for model testing and evaluation"""
    
    def __init__(self, model: Model):
        """
        Initialize Test
        
        Args:
            model: Trained model instance
        """
        self.model = model
        self.predictions = None
        self.metrics = {}
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
        
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model on test data...")
        
        self.predictions = self.model.get_model().predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, self.predictions)
        conf_matrix = confusion_matrix(y_test, self.predictions)
        class_report = classification_report(y_test, self.predictions, output_dict=True)
        
        self.metrics = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, self.predictions)}")
        
        return self.metrics
    
    def get_predictions(self) -> np.ndarray:
        """Get model predictions"""
        return self.predictions


class Forward:
    """Class responsible for forward prediction on new data"""
    
    def __init__(self, model: Model, data_cleaner: Optional[DataCleaner] = None):
       
        self.model = model
        self.data_cleaner = data_cleaner
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data

        arg: X: Feature matrix for prediction
        Returns:
            Array of predictions
        """
        logger.info(f"Making predictions on {X.shape[0]} samples...")
        predictions = self.model.get_model().predict(X)
        logger.info("Predictions completed")
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities
        
        Returns:
            Array (numpy.ndarray) of prediction probabilities
        """
        logger.info(f"Computing prediction probabilities for {X.shape[0]} samples...")
        probabilities = self.model.get_model().predict_proba(X)
        logger.info("Probability computation completed")
        return probabilities
    
    def predict_single(self, features: Dict[str, Any]) -> Tuple[int, float]:
        """
        Make prediction for a single sample
        Returns:
            Tuple of (prediction, probability)
        """
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        prediction = self.predict(X)[0]
        probability = self.predict_proba(X)[0]
        
        return prediction, probability
