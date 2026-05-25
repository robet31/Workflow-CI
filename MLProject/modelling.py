"""
modelling.py (for MLProject / CI Workflow)
Model Training Script for Workflow CI
Author: Ar'raffi Abqori Nur Azizi

Script ini digunakan dalam MLflow Project untuk training otomatis
melalui GitHub Actions CI pipeline.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def load_preprocessed_data(data_dir="winequality_preprocessing"):
    """Load preprocessed train and test data."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=["quality_label"])
    y_train = train_df["quality_label"]
    X_test = test_df.drop(columns=["quality_label"])
    y_test = test_df["quality_label"]
    
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_test, y_pred, save_path="training_confusion_matrix.png"):
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bad Wine", "Good Wine"],
                yticklabels=["Bad Wine", "Good Wine"])
    plt.title("Confusion Matrix - Wine Quality Classification")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def main():
    """Main training function."""
    
    # Parse arguments for hyperparameters (can be passed from MLProject)
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
    
    # Load data
    print("Loading preprocessed data...")
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        # Manual log additional metrics
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        
        # Save confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_artifact(cm_path)
        
        # Save metric info
        metric_info = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": X_train.shape[1]
        }
        
        metric_path = "metric_info.json"
        with open(metric_path, "w") as f:
            json.dump(metric_info, f, indent=2)
        mlflow.log_artifact(metric_path)
        
        # Tags
        mlflow.set_tag("author", "Ar'raffi Abqori Nur Azizi")
        mlflow.set_tag("dataset", "Wine Quality")
        
        run_id = mlflow.active_run().info.run_id
        
        print("\n" + "=" * 50)
        print("MODEL TRAINING RESULTS")
        print("=" * 50)
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"\nRun ID: {run_id}")
        
        # Write run_id to file for CI to pick up
        with open("run_id.txt", "w") as f:
            f.write(run_id)


if __name__ == "__main__":
    main()
