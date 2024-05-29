#!/usr/bin/venv bash
# coding: utf-8

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import tensorflow as tf
from model_funcs import *
from pathlib import Path
from sklearn.metrics import roc_auc_score

import sys
sys.path.append(str(Path(__file__).parents[2]))

if __name__ == "__main__":
    
    root = Path(__file__).parents[2]

    # Load data
    train_data_path = root / "data" / "reduced_0_01" / "train.csv"
    val_data_path = root / "data" / "reduced_0_01" / "val.csv"
    test_data_path = root / "data" / "reduced_0_01" / "test.csv"
    
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)
    test_data = pd.read_csv(test_data_path)

    # Prepare datasets
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_val = val_data.iloc[:, :-1].values
    y_val = val_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Extract feature names for the Random Forest feature importance plot
    feature_names = train_data.columns[:-1]  # Assuming the last column is the target

    # Initialize lists to collect ROC data
    roc_data = []
    model_names = ["Random Forest", "XGBoost", "Neural Network"]

    # Run Random Forest model
    print("Doing hyperparameter tuning and evaluation of Random Forest Model")
    roc_rf, _ = tune_and_evaluate_rf(X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
    roc_data.append(roc_rf)

    # Run XGBoost model
    print("Do XGBoost")
    roc_xgb, _ = evaluate_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
    roc_data.append(roc_xgb)

    """
    # Run Neural Network model
    print("Do Neural Network")
    roc_nn, _ = neural_network_model(X_train, y_train, X_val, y_val, X_test, y_test)
    roc_data.append(roc_nn)
    """
    # Plot and save ROC curves
    plot_roc_curves(roc_data, model_names, filename='roc_curves_comparison.png')
