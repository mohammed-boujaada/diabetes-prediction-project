"""
Machine Learning Models Module
Contains implementations of KNN, Random Forest, and Decision Tree classifiers.
"""

import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class KNNModel:
    """
    K-Nearest Neighbors classifier with hyperparameter optimization.
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto'):
        """
        Initialize KNN model.
        
        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors (default: 5)
        weights : str
            Weight function used in prediction ('uniform' or 'distance')
        algorithm : str
            Algorithm used to compute the nearest neighbors
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm
        )
        self.is_trained = False
        self.best_params = None
    
    def train(self, X_train, y_train):
        """
        Train the KNN model.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        """
        print(f"\n🔧 Training KNN model (k={self.n_neighbors})...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"✓ KNN model trained successfully")
    
    def optimize(self, X_train, y_train, param_grid=None, cv=5):
        """
        Optimize hyperparameters using GridSearchCV.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        param_grid : dict
            Parameter grid for optimization
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict
            Best parameters found
        """
        if param_grid is None:
            param_grid = {
                'n_neighbors': list(range(1, 31, 2)),
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
        
        print(f"\n🔍 Optimizing KNN hyperparameters with GridSearchCV...")
        print(f"   Parameter grid: {param_grid}")
        print(f"   Cross-validation folds: {cv}")
        
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.is_trained = True
        
        print(f"\n✓ Best parameters found: {self.best_params}")
        print(f"  Best F1-score: {grid_search.best_score_:.4f}")
        
        return self.best_params
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : array-like
            Features to predict
            
        Returns:
        --------
        array
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like
            Features to predict
            
        Returns:
        --------
        array
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : array-like
            Test features
        y : array-like
            True labels
            
        Returns:
        --------
        float
            Accuracy score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        return self.model.score(X, y)
    
    def save(self, filepath):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ KNN model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the model file
            
        Returns:
        --------
        KNeighborsClassifier
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ KNN model loaded from {filepath}")
        return model


class RandomForestModel:
    """
    Random Forest classifier with feature importance tracking.
    """
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """
        Initialize Random Forest model.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int
            Maximum depth of the trees
        random_state : int
            Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_importances_ = None
    
    def train(self, X_train, y_train, feature_names=None):
        """
        Train the Random Forest model.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        feature_names : list
            Names of the features
        """
        print(f"\n🔧 Training Random Forest model...")
        print(f"   n_estimators: {self.n_estimators}")
        print(f"   max_depth: {self.max_depth}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_importances_ = self.model.feature_importances_
        
        print(f"✓ Random Forest model trained successfully")
        
        if feature_names:
            self._print_feature_importance(feature_names)
    
    def _print_feature_importance(self, feature_names, top_n=5):
        """
        Print top feature importances.
        
        Parameters:
        -----------
        feature_names : list
            Names of the features
        top_n : int
            Number of top features to display
        """
        importance_df = sorted(
            zip(feature_names, self.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"\n📊 Top {top_n} Feature Importances:")
        for i, (feature, importance) in enumerate(importance_df[:top_n], 1):
            print(f"  {i}. {feature}: {importance:.4f} ({importance*100:.2f}%)")
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importances as a dictionary.
        
        Parameters:
        -----------
        feature_names : list
            Names of the features
            
        Returns:
        --------
        dict
            Feature importances
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importances_))]
        
        return dict(zip(feature_names, self.feature_importances_))
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        """Calculate accuracy score."""
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        return self.model.score(X, y)
    
    def save(self, filepath):
        """Save the trained model to a file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Random Forest model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load a trained model from a file."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Random Forest model loaded from {filepath}")
        return model


class DecisionTreeModel:
    """
    Decision Tree classifier as an interpretable baseline.
    """
    
    def __init__(self, max_depth=10, random_state=42):
        """
        Initialize Decision Tree model.
        
        Parameters:
        -----------
        max_depth : int
            Maximum depth of the tree
        random_state : int
            Random seed for reproducibility
        """
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the Decision Tree model."""
        print(f"\n🔧 Training Decision Tree model...")
        print(f"   max_depth: {self.max_depth}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"✓ Decision Tree model trained successfully")
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        """Calculate accuracy score."""
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        return self.model.score(X, y)
    
    def save(self, filepath):
        """Save the trained model to a file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Decision Tree model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load a trained model from a file."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Decision Tree model loaded from {filepath}")
        return model
