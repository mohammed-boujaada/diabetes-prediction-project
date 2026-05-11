"""
Model Evaluation Module

This module provides functions for evaluating machine learning models.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc
)

logger = logging.getLogger(__name__)

# Set style for better visualizations
sns.set_style("whitegrid")


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        y_pred_proba: Prediction probabilities (optional, for ROC-AUC)
    
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except (ValueError, IndexError):
            logger.warning("Could not calculate ROC-AUC score")
    
    logger.info("Metrics calculated successfully")
    return metrics


def print_classification_report(y_true, y_pred, model_name: str = "Model"):
    """
    Print detailed classification report.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name (str): Name of the model for display
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Classification Report - {model_name}")
    logger.info(f"{'='*60}\n")
    print(classification_report(y_true, y_pred, 
                               target_names=['No Diabetes', 'Diabetes']))


def print_metrics_summary(metrics: dict, model_name: str = "Model"):
    """
    Print summary of evaluation metrics.
    
    Args:
        metrics (dict): Dictionary of metrics
        model_name (str): Name of the model
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Metrics Summary - {model_name}")
    logger.info(f"{'='*60}")
    
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name.upper():.<40} {metric_value:.4f}")
    
    logger.info(f"{'='*60}\n")


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model", 
                          save_path: str = None):
    """
    Plot and display confusion matrix.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name (str): Name of the model
        save_path (str): Path to save the figure (optional)
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'],
                cbar_kws={'label': 'Count'})
    
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curve(y_true, y_pred_proba, model_name: str = "Model",
                   save_path: str = None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True target values
        y_pred_proba: Prediction probabilities
        model_name (str): Name of the model
        save_path (str): Path to save the figure (optional)
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Could not plot ROC curve: {str(e)}")
        return None


def compare_models_metrics(models_results: dict):
    """
    Compare metrics across multiple models.
    
    Args:
        models_results (dict): Dictionary with model names as keys
                              and metrics dicts as values
    
    Returns:
        pd.DataFrame: Comparison dataframe
    """
    comparison_df = pd.DataFrame(models_results).T
    comparison_df = comparison_df.round(4)
    
    logger.info("\nModel Comparison:\n")
    logger.info(comparison_df)
    
    return comparison_df


def plot_models_comparison(comparison_df: pd.DataFrame, save_path: str = None):
    """
    Plot comparison of models across metrics.
    
    Args:
        comparison_df (pd.DataFrame): Comparison dataframe
        save_path (str): Path to save the figure (optional)
    
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    comparison_df.plot(kind='bar', ax=ax)
    
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    
    return fig
