import pandas as pd
from sklearn.model_selection import train_test_split
import os
import joblib

def drop_cols(data, cols_to_drop):
    data = data.drop(columns=cols_to_drop, axis=1)
    return data

def split_data(data, predict_col, test_size=0.33, random_state=52):
    # Splitting data into X and y
    X = data.drop(columns=[predict_col], axis=1)
    y = data[predict_col]

    # Splitting into training, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% train+val, 20% test
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # Of the 80%, 60% train, 20% val

    return X_train, X_val, X_test, y_train, y_val, y_test

def fit_and_save_scaler(data, scaler_filename='models/scaler.pkl'):
    """
    Fit the StandardScaler to the data and save the fitted scaler to a file.

    Args:
    - data: The training data to fit the scaler.
    - scaler_filename: The filename to save the fitted scaler.
    """
    scaler = StandardScaler().fit(data)
    joblib.dump(scaler, scaler_filename)


def apply_standard_scaling(data, scaler_filename='models/scaler.pkl'):
    """
    Load the saved StandardScaler and use it to scale the data.

    Args:
    - data: The data to be scaled.
    - scaler_filename: The filename from where to load the fitted scaler.

    Returns:
    - Scaled data.
    """
    # Load the saved scaler
    scaler = joblib.load(scaler_filename)
    
    # Use the scaler to transform the data
    scaled_data = scaler.transform(data)
    return scaled_data
