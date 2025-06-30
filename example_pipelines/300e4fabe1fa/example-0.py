import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

def load_data(file_path):
    """Load data from CSV file."""
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns")
        return data
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(data):
    """Preprocess data and split into features and target."""
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values detected:")
        print(missing_values[missing_values > 0])
    
    # Create a copy to avoid modifying the original data
    processed_data = data.copy()
    
    # Feature engineering
    
    # 1. BMI categories (underweight, normal, overweight, obese)
    processed_data['BMI_Category'] = pd.cut(
        processed_data['BMI'], 
        bins=[0, 18.5, 25, 30, float('inf')],
        labels=[0, 1, 2, 3]
    ).astype(float)
    
    # 2. Age groups
    processed_data['Age_Group'] = pd.cut(
        processed_data['Age'], 
        bins=[0, 4, 7, 10, float('inf')],
        labels=[0, 1, 2, 3]
    ).astype(float)
    
    # 3. Health score (combination of general, mental, and physical health)
    processed_data['Health_Score'] = (
        (5 - processed_data['GenHlth']) +  # Invert so higher is better
        (30 - processed_data['MentHlth']) / 10 +  # Scale to similar range
        (30 - processed_data['PhysHlth']) / 10  # Scale to similar range
    )
    
    # 4. Risk factors count (high blood pressure, high cholesterol, smoking, etc.)
    risk_factors = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'DiffWalk']
    processed_data['Risk_Factors_Count'] = processed_data[risk_factors].sum(axis=1)
    
    # 5. Healthy lifestyle score (physical activity, fruits, vegetables, etc.)
    healthy_factors = ['PhysActivity', 'Fruits', 'Veggies']
    processed_data['Healthy_Lifestyle_Score'] = processed_data[healthy_factors].sum(axis=1)
    
    print("Added engineered features: BMI_Category, Age_Group, Health_Score, Risk_Factors_Count, Healthy_Lifestyle_Score")
    
    # Split into features and target
    X = processed_data.drop("Diabetes_binary", axis=1)  # Using Diabetes_binary as target
    y = processed_data["Diabetes_binary"]
    
    return X, y

def train_model(X_train, y_train, X_test, y_test):
    """Train and evaluate a logistic regression model."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights to handle imbalance
    # This gives higher weight to the minority class
    class_counts = np.bincount(y_train.astype(int))
    total_samples = len(y_train)
    class_weights = {
        0: total_samples / (2 * class_counts[0]),
        1: total_samples / (2 * class_counts[1])
    }
    print(f"Using class weights to handle imbalance: {class_weights}")
    
    # Train model with class weights
    model = LogisticRegression(
        max_iter=5000, 
        random_state=42,
        class_weight=class_weights
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_train_scaled, X_test_scaled

def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names=None):
    """Evaluate model performance with comprehensive metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, precision_recall_curve,
        average_precision_score
    )
    
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Get probability predictions for ROC and PR curves
    try:
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"Warning: Could not get probability predictions: {e}")
        y_train_prob = None
        y_test_prob = None
    
    # Calculate metrics
    metrics = {
        'Training': {
            'Accuracy': accuracy_score(y_train, y_train_pred),
            'Precision': precision_score(y_train, y_train_pred, zero_division=0),
            'Recall': recall_score(y_train, y_train_pred, zero_division=0),
            'F1 Score': f1_score(y_train, y_train_pred, zero_division=0),
        },
        'Testing': {
            'Accuracy': accuracy_score(y_test, y_test_pred),
            'Precision': precision_score(y_test, y_test_pred, zero_division=0),
            'Recall': recall_score(y_test, y_test_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_test_pred, zero_division=0),
        }
    }
    
    # Add ROC-AUC if probability predictions are available
    if y_train_prob is not None and y_test_prob is not None:
        metrics['Training']['ROC-AUC'] = roc_auc_score(y_train, y_train_prob)
        metrics['Testing']['ROC-AUC'] = roc_auc_score(y_test, y_test_prob)
        
        # Add Average Precision Score (area under PR curve)
        metrics['Training']['Avg Precision'] = average_precision_score(y_train, y_train_prob)
        metrics['Testing']['Avg Precision'] = average_precision_score(y_test, y_test_prob)
    
    # Print metrics
    print("\n=== Model Performance Metrics ===")
    for dataset, dataset_metrics in metrics.items():
        print(f"\n{dataset} Set Metrics:")
        for metric_name, metric_value in dataset_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix (Testing Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    
    # Print classification report
    print("\nClassification Report (Testing Set):")
    print(classification_report(y_test, y_test_pred))
    
    # Feature importance (for logistic regression and tree-based models)
    if hasattr(model, 'coef_') and feature_names is not None:
        print("\nTop 10 Feature Importances (Logistic Regression):")
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(model.coef_[0])
        }).sort_values('Importance', ascending=False)
        print(feature_importance.head(10))
    elif hasattr(model, 'feature_importances_') and feature_names is not None:
        print("\nTop 10 Feature Importances (Tree-based model):")
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(feature_importance.head(10))
    elif hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'):
        print("\nFeature names not provided, cannot display feature importances.")

def save_model(model, scaler, path):
    """Save the trained model and scaler."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'model': model, 'scaler': scaler}, path)
        print(f"Model saved to {path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def compare_models(X_train, y_train, X_test, y_test, sample_size=10000):
    """Compare different models on the dataset."""
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    
    # Create a smaller sample for faster comparison if needed
    if len(X_train) > sample_size:
        print(f"\nUsing {sample_size} samples for model comparison...")
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_sample = X_train.iloc[indices]
        y_train_sample = y_train.iloc[indices]
    else:
        X_train_sample = X_train
        y_train_sample = y_train
    
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=5000, 
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, 
            random_state=42
        )
    }
    
    # Create a scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sample)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate each model
    results = {}
    best_model = None
    best_score = 0
    
    print("\n=== Model Comparison ===")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train_sample)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1': f1
        }
        
        # Print results
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Update best model
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model
    
    # Print summary
    print("\nModel Comparison Summary:")
    print("Model               Accuracy    ROC-AUC     F1 Score")
    print("--------------------------------------------------")
    for name, result in results.items():
        print(f"{name:20} {result['accuracy']:.4f}     {result['roc_auc']:.4f}     {result['f1']:.4f}")
    
    # Return the best model and scaler
    best_name = [name for name, result in results.items() if result['model'] == best_model][0]
    print(f"\nBest model: {best_name} (ROC-AUC: {best_score:.4f})")
    return best_model, scaler

def main():
    # Getting the project root
    project_root = get_project_root()
    
    # Getting the raw data file
    raw_data_file = os.path.join(
        project_root, "datasets", "c99d9bc33649", "c99d9bc33649_b.csv"
    )
    
    # Load data
    data = load_data(raw_data_file)
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Option 1: Train a single model
    print("\n=== Training Logistic Regression Model ===")
    model, scaler, X_train_scaled, X_test_scaled = train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, feature_names=X_train.columns)
    
    # Option 2: Compare different models and select the best one
    print("\n=== Comparing Different Models ===")
    best_model, best_scaler = compare_models(X_train, y_train, X_test, y_test, sample_size=10000)
    
    # Transform data with the best scaler
    X_train_best_scaled = best_scaler.transform(X_train)
    X_test_best_scaled = best_scaler.transform(X_test)
    
    # Evaluate the best model
    print("\n=== Evaluating Best Model ===")
    evaluate_model(best_model, X_train_best_scaled, y_train, X_test_best_scaled, y_test, feature_names=X_train.columns)
    
    # Cross-validation on the best model
    sample_size = min(10000, len(X))  # Use at most 10,000 samples
    print(f"\nPerforming cross-validation on {sample_size} samples...")
    
    # Create a random subset of the data
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample, y_sample = X.iloc[indices], y.iloc[indices]
    
    # Create a pipeline with scaling and the best model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', best_model)
    ])
    
    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X_sample, y_sample, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    # Save the best model
    model_path = os.path.join(project_root, "models", "diabetes_prediction_model.pkl")
    save_model(best_model, best_scaler, model_path)

if __name__ == "__main__":
    main()
