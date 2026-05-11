"""
Data Preprocessing Module

This module handles all data loading, cleaning, and preprocessing operations
for the diabetes prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(filepath: str, delimiter: str = ';') -> pd.DataFrame:
    """
    Load diabetes dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV dataset file
        delimiter (str): CSV delimiter character (default: ';')
    
    Returns:
        pd.DataFrame: Loaded dataset
    
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file cannot be read
    """
    try:
        logger.info(f"Loading dataset from {filepath}")
        df = pd.read_csv(filepath, delimiter=delimiter)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def explore_dataset(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.
    
    Args:
        df (pd.DataFrame): Dataset to explore
    """
    logger.info("Dataset Overview:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"\nColumn names and types:\n{df.dtypes}")
    logger.info(f"\nMissing values:\n{df.isnull().sum()}")
    logger.info(f"\nBasic statistics:\n{df.describe()}")
    if 'Outcome' in df.columns:
        logger.info(f"\nOutcome distribution:\n{df['Outcome'].value_counts()}")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Raw dataset
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    logger.info("Cleaning dataset...")
    
    # Remove duplicates
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df_cleaned)} duplicate rows")
    
    # Remove rows with missing values (if any)
    df_cleaned = df_cleaned.dropna()
    
    logger.info(f"Final dataset shape: {df_cleaned.shape}")
    return df_cleaned


def prepare_features_and_target(df: pd.DataFrame, target_column: str = 'Outcome') -> tuple:
    """
    Separate features (X) and target (y) from the dataset.
    
    Args:
        df (pd.DataFrame): Dataset
        target_column (str): Name of the target column (default: 'Outcome')
    
    Returns:
        tuple: (X features, y target)
    
    Raises:
        ValueError: If target column not found in dataset
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    
    return X, y


def train_test_split_data(X: pd.DataFrame, 
                         y: pd.Series,
                         test_size: float = 0.3,
                         random_state: int = 42) -> tuple:
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of test set (default: 0.3)
        random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Testing set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, 
                   X_test: pd.DataFrame) -> tuple:
    """
    Standardize features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
    
    Returns:
        tuple: (scaled_X_train, scaled_X_test, scaler)
    """
    scaler = StandardScaler()
    
    # Fit on training data, transform both train and test
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    
    logger.info("Features scaled using StandardScaler")
    
    return scaled_X_train, scaled_X_test, scaler


def preprocess_pipeline(filepath: str, 
                       test_size: float = 0.3,
                       random_state: int = 42) -> dict:
    """
    Complete preprocessing pipeline.
    
    Args:
        filepath (str): Path to dataset
        test_size (float): Test set proportion
        random_state (int): Random seed
    
    Returns:
        dict: Dictionary containing all preprocessed data and objects
    """
    # Load and clean data
    df = load_dataset(filepath)
    explore_dataset(df)
    df_cleaned = clean_dataset(df)
    
    # Prepare features and target
    X, y = prepare_features_and_target(df_cleaned)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaled_X_train, scaled_X_test, scaler = scale_features(X_train, X_test)
    
    return {
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaled_X_train': scaled_X_train,
        'scaled_X_test': scaled_X_test,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }
