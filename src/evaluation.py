"""
Evaluation Module
Contains functions for model evaluation, metrics calculation, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like, optional
        Predicted probabilities (for ROC-AUC)
        
    Returns:
    --------
    dict
        Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        # For binary classification, use probability of positive class
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    return metrics


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print detailed classification report.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    target_names : list, optional
        Names of the classes
    """
    if target_names is None:
        target_names = ['Class 0', 'Class 1']
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=target_names))


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', 
                         normalize=False, save_path=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    title : str
        Plot title
    normalize : bool
        Whether to normalize the confusion matrix
    save_path : str, optional
        Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title += ' (Normalized)'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_proba, title='ROC Curve', save_path=None):
    """
    Plot ROC curve for binary classification.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities for positive class
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    # Handle 2D probability array
    if len(y_proba.shape) > 1:
        y_proba = y_proba[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")
    
    plt.show()


def plot_multiple_roc_curves(models_dict, X_test, y_test, 
                            title='ROC Curves Comparison', save_path=None):
    """
    Plot ROC curves for multiple models on the same figure.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: model_object}
    X_test : array-like
        Test features
    y_test : array-like
        True test labels
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_dict)))
    
    for (model_name, model), color in zip(models_dict.items(), colors):
        y_proba = model.predict_proba(X_test)
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{model_name} (AUC = {auc_score:.4f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curves comparison saved to {save_path}")
    
    plt.show()


def plot_feature_importance(feature_importances, feature_names, 
                           title='Feature Importance', top_n=10, save_path=None):
    """
    Plot feature importance bar chart.
    
    Parameters:
    -----------
    feature_importances : array-like or dict
        Feature importance values
    feature_names : list
        Names of the features
    title : str
        Plot title
    top_n : int
        Number of top features to display
    save_path : str, optional
        Path to save the figure
    """
    # Convert dict to array if needed
    if isinstance(feature_importances, dict):
        feature_names = list(feature_importances.keys())
        feature_importances = list(feature_importances.values())
    
    # Sort and get top N
    indices = np.argsort(feature_importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), 
             [feature_importances[i] for i in indices][::-1],
             color='steelblue', alpha=0.8)
    plt.yticks(range(len(indices)), 
               [feature_names[i] for i in indices][::-1])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved to {save_path}")
    
    plt.show()


def compare_models_metrics(models_dict, X_test, y_test, save_path=None):
    """
    Compare metrics across multiple models and return a DataFrame.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: model_object}
    X_test : array-like
        Test features
    y_test : array-like
        True test labels
    save_path : str, optional
        Path to save the CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing metrics for all models
    """
    results = []
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        metrics['model'] = model_name
        
        results.append(metrics)
        
        print(f"\n📊 {model_name}:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    df_results = pd.DataFrame(results)
    df_results = df_results[['model', 'accuracy', 'precision', 'recall', 
                             'f1_score', 'roc_auc']]
    
    if save_path:
        df_results.to_csv(save_path, index=False)
        print(f"\n✓ Model comparison saved to {save_path}")
    
    return df_results


def plot_models_comparison_bar(df_metrics, save_path=None):
    """
    Plot bar chart comparing metrics across models.
    
    Parameters:
    -----------
    df_metrics : pd.DataFrame
        DataFrame with model metrics
    save_path : str, optional
        Path to save the figure
    """
    metrics_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    available_metrics = [m for m in metrics_cols if m in df_metrics.columns]
    
    df_plot = df_metrics.set_index('model')[available_metrics]
    
    plt.figure(figsize=(12, 6))
    df_plot.plot(kind='bar', figsize=(12, 6), alpha=0.8)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison bar chart saved to {save_path}")
    
    plt.show()
