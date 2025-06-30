# Diabetes Prediction Model

This repository contains a machine learning pipeline for predicting diabetes based on various health indicators.

## Project Structure

- `example_pipelines/300e4fabe1fa/example-0.py`: Main script for data processing, model training, and evaluation
- `example_pipelines/utils.py`: Utility functions for the project
- `datasets/`: Contains the dataset used for training
- `models/`: Directory for saving trained models

## Features

- Data loading and preprocessing with error handling
- Feature engineering to create additional predictive features
- Feature scaling using StandardScaler
- Class imbalance handling with weighted classes
- Multiple model comparison (Logistic Regression, Random Forest, Gradient Boosting)
- Comprehensive model evaluation with multiple metrics
- Feature importance analysis
- Cross-validation on a sample of the data for efficiency
- Model persistence for future use

## Requirements

See `requirements.txt` for the list of dependencies.

## Usage

To run the pipeline:

```bash
python example_pipelines/300e4fabe1fa/example-0.py
```

This will:
1. Load the dataset
2. Preprocess the data and create engineered features
3. Train multiple models (Logistic Regression, Random Forest, Gradient Boosting)
4. Compare models and select the best one
5. Evaluate the models with comprehensive metrics
6. Perform cross-validation
7. Save the best model to the `models/` directory

## Model Performance

The initial logistic regression model achieves approximately 87% accuracy on the test set, with the following metrics:
- Precision: 0.88 for class 0, 0.55 for class 1
- Recall: 0.98 for class 0, 0.17 for class 1
- F1-score: 0.93 for class 0, 0.25 for class 1
- ROC-AUC: ~0.83

The enhanced pipeline compares multiple models and typically achieves better performance, especially with tree-based models like Random Forest and Gradient Boosting.

## Improvements Made

The following improvements were made to the original script:

### Code Structure and Organization
1. Fixed project root path detection
2. Refactored code into modular functions with proper docstrings
3. Added comprehensive error handling
4. Implemented model persistence with both model and scaler

### Data Processing
1. Corrected target variable selection (Diabetes_binary instead of Income)
2. Added data validation and missing value detection
3. Implemented feature engineering to create additional predictive features:
   - BMI categories
   - Age groups
   - Health score
   - Risk factors count
   - Healthy lifestyle score

### Model Training and Evaluation
1. Added class imbalance handling with weighted classes
2. Implemented multiple model comparison (Logistic Regression, Random Forest, Gradient Boosting)
3. Enhanced model evaluation with comprehensive metrics:
   - Accuracy, Precision, Recall, F1-score
   - ROC-AUC and Average Precision
   - Confusion matrix
   - Feature importance analysis
4. Optimized cross-validation for large datasets
5. Added pipeline for consistent preprocessing in cross-validation

### Documentation
1. Added detailed docstrings to all functions
2. Created comprehensive README with usage instructions
3. Added comments explaining complex operations
