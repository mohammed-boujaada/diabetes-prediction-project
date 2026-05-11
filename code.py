"""
Example usage of the Diabetes Prediction Project

This script demonstrates how to use the trained models and modules
for making predictions on new data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pickle
import pandas as pd
from data_preprocessing import preprocess_pipeline
from models import KNNModel, RandomForestModel
from evaluation import calculate_metrics, print_classification_report, print_metrics_summary


def example_1_train_from_scratch():
    """Example 1: Train models from scratch"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Training Models from Scratch")
    print("="*70 + "\n")
    
    # Preprocess data
    print("Step 1: Preprocessing data...")
    prep_data = preprocess_pipeline('dataset/dataset_diabet.csv')
    
    # Train KNN
    print("\nStep 2: Training KNN model...")
    knn = KNNModel(n_neighbors=16)
    knn.train(prep_data['scaled_X_train'], prep_data['y_train'])
    knn_pred = knn.predict(prep_data['scaled_X_test'])
    knn_metrics = calculate_metrics(prep_data['y_test'], knn_pred)
    
    print(f"\nKNN Results:")
    print(f"  Accuracy: {knn_metrics['accuracy']:.4f}")
    print(f"  Precision: {knn_metrics['precision']:.4f}")
    print(f"  Recall: {knn_metrics['recall']:.4f}")
    print(f"  F1-Score: {knn_metrics['f1']:.4f}")
    
    # Train Random Forest
    print("\nStep 3: Training Random Forest model...")
    rf = RandomForestModel(n_estimators=100, random_state=42)
    rf.train(prep_data['X_train'], prep_data['y_train'])
    rf_pred = rf.predict(prep_data['X_test'])
    rf_metrics = calculate_metrics(prep_data['y_test'], rf_pred)
    
    print(f"\nRandom Forest Results:")
    print(f"  Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"  Precision: {rf_metrics['precision']:.4f}")
    print(f"  Recall: {rf_metrics['recall']:.4f}")
    print(f"  F1-Score: {rf_metrics['f1']:.4f}")
    
    return prep_data, knn, rf


def example_2_load_trained_models():
    """Example 2: Load pre-trained models and make predictions"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Using Pre-trained Models")
    print("="*70 + "\n")
    
    # Load preprocessed data
    prep_data = preprocess_pipeline('dataset/dataset_diabet.csv')
    
    # Load trained models
    models_dir = 'results'
    print("Loading trained models...")
    
    with open(os.path.join(models_dir, 'knn_optimized.pkl'), 'rb') as f:
        knn_model = pickle.load(f)
    
    with open(os.path.join(models_dir, 'random_forest.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    
    print("✓ Models loaded successfully!\n")
    
    # Make predictions
    print("Making predictions...")
    knn_pred = knn_model.predict(prep_data['X_test'])
    rf_pred = rf_model.predict(prep_data['X_test'])
    
    # Evaluate
    knn_metrics = calculate_metrics(prep_data['y_test'], knn_pred)
    rf_metrics = calculate_metrics(prep_data['y_test'], rf_pred)
    
    print(f"\nKNN Performance: Accuracy={knn_metrics['accuracy']:.4f}, F1={knn_metrics['f1']:.4f}")
    print(f"Random Forest Performance: Accuracy={rf_metrics['accuracy']:.4f}, F1={rf_metrics['f1']:.4f}")
    
    return prep_data, knn_model, rf_model


def example_3_predict_single_patient():
    """Example 3: Make predictions for a single patient"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Single Patient Prediction")
    print("="*70 + "\n")
    
    # Load trained model
    with open('results/knn_optimized.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    
    with open('results/random_forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    # Create patient data
    patient_data = pd.DataFrame({
        'Pregnancies': [2],
        'Glucose': [150],
        'BloodPressure': [70],
        'SkinThickness': [30],
        'Insulin': [0],
        'BMI': [30],
        'DiabetesPedigreeFunction': [0.5],
        'Age': [45]
    })
    
    print("Patient Information:")
    print(patient_data.to_string(index=False))
    
    # Make predictions
    print("\n" + "-"*70)
    print("PREDICTIONS")
    print("-"*70 + "\n")
    
    # KNN prediction
    knn_pred = knn_model.predict(patient_data)
    knn_proba = knn_model.predict_proba(patient_data)
    
    print("KNN (Optimized) Model:")
    print(f"  Prediction: {'DIABETES RISK DETECTED' if knn_pred[0] == 1 else 'NO DIABETES'}")
    print(f"  Confidence: {max(knn_proba[0]):.2%}")
    print(f"  Diabetes Probability: {knn_proba[0][1]:.2%}")
    
    # Random Forest prediction
    rf_pred = rf_model.predict(patient_data)
    rf_proba = rf_model.predict_proba(patient_data)
    
    print("\nRandom Forest Model:")
    print(f"  Prediction: {'DIABETES RISK DETECTED' if rf_pred[0] == 1 else 'NO DIABETES'}")
    print(f"  Confidence: {max(rf_proba[0]):.2%}")
    print(f"  Diabetes Probability: {rf_proba[0][1]:.2%}")
    
    # Ensemble decision
    ensemble_pred = (knn_pred[0] + rf_pred[0]) / 2
    print("\nEnsemble Decision (Average):")
    print(f"  Score: {ensemble_pred:.2f}")
    print(f"  Recommendation: {'CONSULT HEALTHCARE PROVIDER' if ensemble_pred >= 0.5 else 'NO IMMEDIATE CONCERN'}")


def example_4_compare_models():
    """Example 4: Train and compare multiple models"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Model Comparison")
    print("="*70 + "\n")
    
    from models import DecisionTreeModel
    
    # Preprocess
    prep_data = preprocess_pipeline('dataset/dataset_diabet.csv')
    
    models = {}
    results = {}
    
    # Train KNN
    print("Training KNN...")
    knn = KNNModel(n_neighbors=16)
    knn.train(prep_data['scaled_X_train'], prep_data['y_train'])
    knn_pred = knn.predict(prep_data['scaled_X_test'])
    results['KNN'] = calculate_metrics(prep_data['y_test'], knn_pred)
    models['KNN'] = knn
    
    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestModel()
    rf.train(prep_data['X_train'], prep_data['y_train'])
    rf_pred = rf.predict(prep_data['X_test'])
    results['Random Forest'] = calculate_metrics(prep_data['y_test'], rf_pred)
    models['Random Forest'] = rf
    
    # Train Decision Tree
    print("Training Decision Tree...")
    dt = DecisionTreeModel()
    dt.train(prep_data['X_train'], prep_data['y_train'])
    dt_pred = dt.predict(prep_data['X_test'])
    results['Decision Tree'] = calculate_metrics(prep_data['y_test'], dt_pred)
    models['Decision Tree'] = dt
    
    # Display comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70 + "\n")
    
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.to_string())
    
    # Find best model
    best_model = comparison_df['f1'].idxmax()
    print(f"\n✓ Best Model: {best_model} (F1-Score: {comparison_df.loc[best_model, 'f1']:.4f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Diabetes Prediction Project Examples"
    )
    parser.add_argument(
        '--example',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help='Which example to run (0=all, 1-4=specific)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.example == 0 or args.example == 1:
            example_1_train_from_scratch()
        
        if args.example == 0 or args.example == 2:
            example_2_load_trained_models()
        
        if args.example == 0 or args.example == 3:
            example_3_predict_single_patient()
        
        if args.example == 0 or args.example == 4:
            example_4_compare_models()
        
        print("\n" + "="*70)
        print("✓ Examples completed successfully!")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you're in the project root directory")
        print("2. Check that dataset/dataset_diabet.csv exists")
        print("3. Install requirements: pip install -r requirements.txt")
        print("4. Run Jupyter notebooks for detailed analysis")
