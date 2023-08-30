import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

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

