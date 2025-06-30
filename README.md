# Diabetes Prediction Model

This repository contains a machine learning pipeline for predicting diabetes based on various health indicators.

## Project Structure

- `example_pipelines/300e4fabe1fa/example-0.py`: Main script for data processing, model training, and evaluation
- `example_pipelines/utils.py`: Utility functions for the project
- `datasets/`: Contains the dataset used for training
- `models/`: Directory for saving trained models

## Features

- Data loading and preprocessing
- Feature scaling using StandardScaler
- Logistic Regression model for binary classification
- Model evaluation with multiple metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- Cross-validation on a sample of the data
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
2. Preprocess the data
3. Train a logistic regression model
4. Evaluate the model
5. Perform cross-validation
6. Save the trained model to the `models/` directory

## Model Performance

The model achieves approximately 87% accuracy on the test set, with the following metrics:
- Precision: 0.88 for class 0, 0.55 for class 1
- Recall: 0.98 for class 0, 0.17 for class 1
- F1-score: 0.93 for class 0, 0.25 for class 1
- ROC-AUC: ~0.83

## Improvements Made

The following improvements were made to the original script:
1. Fixed project root path detection
2. Corrected target variable selection (Diabetes_binary instead of Income)
3. Added data preprocessing and feature scaling
4. Implemented comprehensive model evaluation
5. Added cross-validation on a sample of the data
6. Added error handling for file operations and model training
7. Implemented model persistence
8. Refactored code into modular functions
9. Added documentation
