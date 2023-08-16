import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score


def preprocess_data(data, test_size=0.33, random_state=52):
    """
    Preprocesses the given dataset by splitting it into training and test sets, and then scaling the features.

    Parameters:
    - data (DataFrame): The input dataset with features in all columns except the last one, which contains the target variable.
    - test_size (float, optional): Proportion of the dataset to be used as the test set. Default is 0.33.
    - random_state (int, optional): The seed used by the random number generator for reproducibility. Default is 52.

    Returns:
    - X_train_scaled (array): Scaled features for the training set.
    - X_test_scaled (array): Scaled features for the test set.
    - y_train (Series): Target variable for the training set.
    - y_test (Series): Target variable for the test set.
    """

    # Split Data into X and y
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    print("Shape of X: {info}".format(info = X.shape))
    print("Shape of y: {info}".format(info = y.shape))
    
    # Splitting into training, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% train+val, 20% test
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # Of the 80%, 60% train, 20% val
    # Scale X dataset
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model(data):
    
    """
    Build a regression Sequential model based on the shape of the input data.

    Parameters:
    - data: Input data used to infer the shape for the model's input layer.

    Returns:
    - A compiled Keras Sequential model ready for training.
    """

    # Assuming the data is 2D with shape (num_samples, num_features)
    input_dim = data.shape[1]

    # Define the Sequential model for regression
    model = tf.keras.Sequential([
        # Add a Dense layer with 128 units and ReLU activation
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        # Add another Dense layer with 64 units and ReLU activation
        layers.Dense(64, activation='relu'),
        # Add the output layer for regression
        layers.Dense(1)
    ])

    # Create Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Compile the model (prepare for training)
    model.compile(optimizer=optimizer, 
                  loss='mean_squared_error',
                  metrics=['mae'])

    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    """
    Train the provided Keras model using the given data.

    Parameters:
    - model: Keras model to be trained.
    - X_train, y_train: Training data and corresponding labels (or target values).
    - X_val, y_val (optional): Validation data and corresponding labels. 
                               Used to evaluate model's performance on unseen data during training.
    - epochs (default=100): Number of times the learning algorithm will work through the entire training dataset.
    - batch_size (default=32): Number of samples per gradient update.

    Returns:
    - History object. Its History.history attribute contains all information about the training history.
    """

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stop])


    return history

def evaluate_model_performance(model, X_test, y_test, verbose=True):
    """
    Evaluates the performance of a trained TensorFlow model.

    Parameters:
    - model: A trained TensorFlow model.
    - X_test: Test data features.
    - y_test: True labels for the test data.
    - verbose: If True, print the performance metrics.

    Returns:
    - mse: Mean Squared Error on the test set.
    - mae: Mean Absolute Error on the test set.
    """
    # Compute predictions for the test set
    y_pred = model.predict(X_test)
    
    # Compute the MSE and MAE metrics
    mse = tf.keras.losses.MeanSquaredError()(y_test, y_pred).numpy()
    mae = tf.keras.losses.MeanAbsoluteError()(y_test, y_pred).numpy()

    # Optionally print the results
    if verbose:
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")

    return mse, mae




# Example Usage
if __name__ == "__main__":

    #Load in the data
    data = pd.read_csv("/data/admissions_data.csv")

    # Pre process the data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data)

    # Build and train the model
    model = build_model(X_train)
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=16)

    #Optionally, to visualize training history (requires matplotlib)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()


    # Run prediction on test data and print results
    mse, mae = evaluate_model_performance(model, X_test, y_test)