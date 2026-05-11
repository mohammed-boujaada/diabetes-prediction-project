# Diabetes Prediction Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

A comprehensive machine learning project for predicting diabetes risk using multiple classification models. This project demonstrates best practices in data preprocessing, model training, evaluation, and visualization.

## 📋 Project Overview

This project was developed as a final year project during my Bachelor's degree in Software Engineering and represents professional-grade machine learning practices. After extensive experimentation with multiple models, **KNN (K-Nearest Neighbors)** and **Random Forest** were identified as the best performing algorithms.

### Key Features
- ✅ Comprehensive exploratory data analysis (EDA)
- ✅ Multiple ML models (KNN, Random Forest, Decision Tree)
- ✅ Hyperparameter optimization (GridSearchCV)
- ✅ Detailed performance evaluation and comparison
- ✅ Professional Jupyter notebooks with clear documentation
- ✅ Production-ready Python package structure
- ✅ Visualization and reporting capabilities

## 📊 Project Structure

```
diabetes-prediction-project/
│
├── dataset/
│   └── dataset_diabet.csv              # Main diabetes dataset
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA and data analysis
│   ├── 02_model_training.ipynb         # Model training and optimization
│   └── 03_results_analysis.ipynb       # Results visualization and analysis
│
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── data_preprocessing.py           # Data loading and preprocessing
│   ├── models.py                       # ML model implementations
│   └── evaluation.py                   # Evaluation metrics and visualization
│
├── results/
│   ├── knn_optimized.pkl              # Trained KNN model (optimized)
│   ├── random_forest.pkl              # Trained Random Forest model
│   ├── decision_tree.pkl              # Trained Decision Tree model
│   ├── scaler.pkl                     # StandardScaler for feature scaling
│   ├── model_comparison.csv           # Model performance metrics
│   └── detailed_metrics.csv           # Detailed evaluation metrics
│
├── images/
│   ├── confusion_matrices.png         # Confusion matrices visualization
│   ├── roc_curves.png                 # ROC curves for all models
│   └── feature_importance.png         # Feature importance plot
│
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup configuration
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

## 🎯 Dataset Information

### Dataset: Pima Indians Diabetes Database

**Source**: UCI Machine Learning Repository

**Characteristics**:
- **Samples**: 768 records
- **Features**: 8 clinical measurements
- **Target**: Binary classification (0: No Diabetes, 1: Diabetes)
- **Class Distribution**: ~35% positive (diabetic), ~65% negative (non-diabetic)

### Features

| Feature | Description | Unit |
|---------|-------------|------|
| Pregnancies | Number of pregnancies | Count |
| Glucose | Plasma glucose concentration | mg/dL |
| BloodPressure | Diastolic blood pressure | mmHg |
| SkinThickness | Triceps skin fold thickness | mm |
| Insulin | 2-Hour serum insulin | mu U/ml |
| BMI | Body mass index | kg/m² |
| DiabetesPedigreeFunction | Diabetes pedigree function | Score |
| Age | Age of the individual | Years |
| **Outcome** | **Diabetes (1) or Not (0)** | **Binary** |

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction-project.git
   cd diabetes-prediction-project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n diabetes python=3.9
   conda activate diabetes
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import pandas; import sklearn; print('Installation successful!')"
   ```

## 🚀 Quick Start

### Running Jupyter Notebooks

```bash
# Navigate to project directory
cd diabetes-prediction-project

# Start Jupyter
jupyter notebook

# Open the notebooks in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_model_training.ipynb
# 3. notebooks/03_results_analysis.ipynb
```

### Using the Package Programmatically

```python
import sys
sys.path.insert(0, 'src')

from data_preprocessing import preprocess_pipeline
from models import KNNModel, RandomForestModel
from evaluation import calculate_metrics

# Prepare data
prep_data = preprocess_pipeline('dataset/dataset_diabet.csv')

# Train KNN model
knn = KNNModel(n_neighbors=5)
knn.train(prep_data['scaled_X_train'], prep_data['y_train'])

# Make predictions
predictions = knn.predict(prep_data['scaled_X_test'])

# Evaluate
metrics = calculate_metrics(prep_data['y_test'], predictions)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## 📈 Models and Results

### Models Tested

1. **K-Nearest Neighbors (KNN)**
   - Simple, effective baseline
   - Hyperparameter: number of neighbors (k)
   - Optimization: GridSearchCV with k ∈ [1, 30]

2. **Random Forest**
   - Ensemble method
   - Provides feature importance
   - Strong generalization
   - Parameters: 100 estimators, max_depth=10

3. **Decision Tree**
   - Interpretable baseline
   - Fast training and prediction
   - Parameters: max_depth=10

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **KNN (Optimized)** | **0.7662** | **0.6923** | **0.5686** | **0.6250** | **0.7435** |
| Random Forest | 0.7597 | 0.6667 | 0.5686 | 0.6122 | 0.7342 |
| Decision Tree | 0.7143 | 0.6000 | 0.5098 | 0.5517 | 0.6537 |
| KNN (k=5) | 0.7273 | 0.6207 | 0.5490 | 0.5816 | 0.7102 |

**Key Finding**: The optimized KNN model (k=16) achieved the best F1-score of 0.6250, balancing precision and recall effectively.

### Feature Importance (Random Forest)

Top 5 most important features:
1. **Glucose** - 0.2845 (28.5%)
2. **BMI** - 0.2154 (21.5%)
3. **DiabetesPedigreeFunction** - 0.1547 (15.5%)
4. **Age** - 0.1421 (14.2%)
5. **BloodPressure** - 0.0896 (9.0%)

## 📊 Workflow

### 1. Data Exploration (Notebook 01)
- Load and inspect dataset
- Check data quality (missing values, duplicates)
- Analyze feature distributions
- Examine correlations
- Visualize class balance
- Identify key patterns

### 2. Data Preprocessing
- Handle missing values
- Remove duplicates
- Feature scaling (StandardScaler)
- Train-test split (70-30)
- Prepare data for modeling

### 3. Model Training (Notebook 02)
- Train multiple models
- Optimize hyperparameters using GridSearchCV
- Evaluate on test set
- Compare model performance
- Save best models

### 4. Results Analysis (Notebook 03)
- Generate confusion matrices
- Plot ROC curves
- Calculate detailed metrics
- Analyze prediction probabilities
- Provide recommendations

## 🔍 Detailed Usage

### Making Predictions with Trained Model

```python
import pickle
import pandas as pd
from data_preprocessing import load_dataset, prepare_features_and_target

# Load the trained model and scaler
with open('results/knn_optimized.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare new patient data
new_patient = pd.DataFrame({
    'Pregnancies': [2],
    'Glucose': [150],
    'BloodPressure': [70],
    'SkinThickness': [30],
    'Insulin': [0],
    'BMI': [30],
    'DiabetesPedigreeFunction': [0.5],
    'Age': [45]
})

# Make prediction
prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)

print(f"Prediction: {'Diabetes' if prediction[0] == 1 else 'No Diabetes'}")
print(f"Probability: {probability[0][1]:.2%}")
```

## 📚 Module Documentation

### `data_preprocessing.py`
- `load_dataset()` - Load CSV dataset with custom delimiter
- `explore_dataset()` - Print dataset information
- `clean_dataset()` - Remove duplicates and missing values
- `prepare_features_and_target()` - Separate X and y
- `train_test_split_data()` - Split data with reproducibility
- `scale_features()` - Standardize features
- `preprocess_pipeline()` - Complete preprocessing workflow

### `models.py`
Classes for ML models:
- `KNNModel` - K-Nearest Neighbors with optimization
- `RandomForestModel` - Random Forest with feature importance
- `DecisionTreeModel` - Decision Tree baseline

### `evaluation.py`
- `calculate_metrics()` - Compute accuracy, precision, recall, F1, ROC-AUC
- `print_classification_report()` - Detailed sklearn report
- `plot_confusion_matrix()` - Visualize confusion matrix
- `plot_roc_curve()` - Plot ROC curve
- `compare_models_metrics()` - Compare multiple models

## 🎓 Educational Value

This project demonstrates:
- ✅ Professional Python package structure
- ✅ Comprehensive data analysis pipeline
- ✅ Multiple ML algorithms comparison
- ✅ Hyperparameter optimization techniques
- ✅ Best practices in model evaluation
- ✅ Clear documentation and code comments
- ✅ Reproducible results (fixed random_state)
- ✅ Visualization best practices

## 💡 Key Insights

1. **Glucose levels** are the strongest predictor of diabetes
2. **BMI** is the second most important factor
3. The **optimized KNN model** provides the best balance between precision and recall
4. Class imbalance (35% positive) doesn't significantly impact model performance
5. All models achieve **>71% accuracy** on the test set

## 🔬 Technologies Used

- **Python 3.8+** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **Jupyter Notebook** - Interactive development environment

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@project{diabetes_prediction_2024,
  title={Diabetes Prediction Using Machine Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/diabetes-prediction-project}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests
- Share insights

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👨‍🎓 About

**Author**: Your Name  
**Education**: Master's Student in Artificial Intelligence and Emerging Technologies  
**Background**: Bachelor's degree in Software Engineering  
**Original Project**: Final year capstone project

## 📧 Contact

- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Scikit-learn for excellent ML libraries
- The open-source community for tools and inspiration

---

**Last Updated**: January 2024  
**Status**: Active & Maintained  
**Version**: 1.0.0

⭐ If you found this project useful, please consider giving it a star!
