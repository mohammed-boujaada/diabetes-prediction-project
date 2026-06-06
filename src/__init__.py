"""
Diabetes Prediction ML Project
Professional Python package for diabetes prediction using various ML models.
"""

__version__ = "1.0.0"
__author__ = "Boujaada Mohammed"
__email__ = "mohammedboujaada51@gmail.com"
__description__ = "Machine Learning models for diabetes prediction"
__license__ = "Gl"
__url__ = "https://github.com/MohammedBoujaada/diabetes-prediction-project"

# Import main classes for easier access
from .data_preprocessing import preprocess_pipeline
from .models import KNNModel, RandomForestModel, DecisionTreeModel
from .evaluation import calculate_metrics

__all__ = [
    'preprocess_pipeline',
    'KNNModel',
    'RandomForestModel', 
    'DecisionTreeModel',
    'calculate_metrics'
]
