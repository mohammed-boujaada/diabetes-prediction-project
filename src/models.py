"""
Machine Learning Models Module

This module implements various machine learning models for diabetes prediction.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class KNNModel:
    """
    K-Nearest Neighbors model for diabetes prediction.
    
    Attributes:
        model: Trained KNN classifier
        best_k: Best number of neighbors found during optimization
        pipeline: Complete preprocessing + model pipeline
    """
    
    def __init__(self, n_neighbors: int = 5):
        """
        Initialize KNN model.
        
        Args:
            n_neighbors (int): Number of neighbors to use (default: 5)
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.best_k = n_neighbors
        self.pipeline = None
        logger.info(f"KNN model initialized with n_neighbors={n_neighbors}")
    
    def train(self, X_train, y_train):
        """
        Train the KNN model.
        
        Args:
            X_train: Training features (should be scaled)
            y_train: Training target variable
        
        Returns:
            self: Fitted model
        """
        self.model.fit(X_train, y_train)
        logger.info("KNN model trained successfully")
        return self
    
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test: Test features (should be scaled)
        
        Returns:
            np.array: Predictions
        """
        predictions = self.model.predict(X_test)
        return predictions
    
    def predict_proba(self, X_test):
        """
        Get prediction probabilities.
        
        Args:
            X_test: Test features (should be scaled)
        
        Returns:
            np.array: Prediction probabilities
        """
        return self.model.predict_proba(X_test)
    
    def optimize_k(self, X_train, y_train, X_test, y_test, k_range: range):
        """
        Find the optimal number of neighbors using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            k_range (range): Range of k values to test
        
        Returns:
            dict: Results with best k and scores
        """
        logger.info(f"Optimizing KNN with k range {k_range}")
        
        # Create pipeline
        scaler = StandardScaler()
        knn = KNeighborsClassifier()
        pipeline = Pipeline([('scaler', scaler), ('knn', knn)])
        
        # Grid search
        param_grid = {'knn__n_neighbors': list(k_range)}
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        self.best_k = grid_search.best_params_['knn__n_neighbors']
        self.pipeline = grid_search.best_estimator_
        self.model = grid_search.best_estimator_.named_steps['knn']
        
        logger.info(f"Best k value: {self.best_k}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return {
            'best_k': self.best_k,
            'best_score': grid_search.best_score_,
            'test_score': grid_search.score(X_test, y_test),
            'all_results': grid_search.cv_results_
        }


class RandomForestModel:
    """
    Random Forest model for diabetes prediction.
    
    Attributes:
        model: Trained Random Forest classifier
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42, **kwargs):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators (int): Number of trees (default: 100)
            random_state (int): Random seed (default: 42)
            **kwargs: Additional parameters for RandomForestClassifier
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )
        logger.info(f"Random Forest model initialized with {n_estimators} estimators")
    
    def train(self, X_train, y_train):
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target variable
        
        Returns:
            self: Fitted model
        """
        self.model.fit(X_train, y_train)
        logger.info("Random Forest model trained successfully")
        return self
    
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test: Test features
        
        Returns:
            np.array: Predictions
        """
        predictions = self.model.predict(X_test)
        return predictions
    
    def predict_proba(self, X_test):
        """
        Get prediction probabilities.
        
        Args:
            X_test: Test features
        
        Returns:
            np.array: Prediction probabilities
        """
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            feature_names (list): List of feature names
        
        Returns:
            pd.DataFrame: Feature importance dataframe sorted by importance
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Feature importance retrieved")
        return importance_df


class DecisionTreeModel:
    """
    Decision Tree model for diabetes prediction.
    
    Attributes:
        model: Trained Decision Tree classifier
    """
    
    def __init__(self, random_state: int = 42, **kwargs):
        """
        Initialize Decision Tree model.
        
        Args:
            random_state (int): Random seed (default: 42)
            **kwargs: Additional parameters for DecisionTreeClassifier
        """
        self.model = DecisionTreeClassifier(random_state=random_state, **kwargs)
        logger.info("Decision Tree model initialized")
    
    def train(self, X_train, y_train):
        """
        Train the Decision Tree model.
        
        Args:
            X_train: Training features
            y_train: Training target variable
        
        Returns:
            self: Fitted model
        """
        self.model.fit(X_train, y_train)
        logger.info("Decision Tree model trained successfully")
        return self
    
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test: Test features
        
        Returns:
            np.array: Predictions
        """
        predictions = self.model.predict(X_test)
        return predictions
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            feature_names (list): List of feature names
        
        Returns:
            pd.DataFrame: Feature importance dataframe sorted by importance
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Feature importance retrieved")
        return importance_df
