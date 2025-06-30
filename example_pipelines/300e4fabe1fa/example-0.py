import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

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
    
    # Split into features and target
    X = data.drop("Diabetes_binary", axis=1)  # Using Diabetes_binary as target
    y = data["Diabetes_binary"]
    
    return X, y

def train_model(X_train, y_train, X_test, y_test):
    """Train and evaluate a logistic regression model."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_train_scaled, X_test_scaled

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance."""
    # Training performance
    train_score = model.score(X_train, y_train)
    print(f"Training accuracy: {train_score:.4f}")
    
    # Test performance
    y_pred = model.predict(X_test)
    test_score = model.score(X_test, y_test)
    print(f"Test accuracy: {test_score:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC-AUC for binary classification
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC: {roc_auc:.4f}")
    except Exception as e:
        print(f"Could not calculate ROC-AUC score: {e}")

def save_model(model, scaler, path):
    """Save the trained model and scaler."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'model': model, 'scaler': scaler}, path)
        print(f"Model saved to {path}")
    except Exception as e:
        print(f"Error saving model: {e}")

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
    
    # Train model
    model, scaler, X_train_scaled, X_test_scaled = train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Cross-validation on a smaller subset for efficiency
    # Using 10% of the data for cross-validation to speed up the process
    sample_size = min(10000, len(X))  # Use at most 10,000 samples
    print(f"\nPerforming cross-validation on {sample_size} samples...")
    
    # Create a random subset of the data
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample, y_sample = X.iloc[indices], y.iloc[indices]
    
    # Perform cross-validation on the subset
    cv_scores = cross_val_score(
        LogisticRegression(max_iter=5000, random_state=42), 
        X_sample, y_sample, 
        cv=5
    )
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    # Save model
    model_path = os.path.join(project_root, "models", "diabetes_prediction_model.pkl")
    save_model(model, scaler, model_path)

if __name__ == "__main__":
    main()
