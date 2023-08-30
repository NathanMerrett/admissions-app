import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

# Local module imports
from modules.preprocess import drop_cols, split_data
from modules.model import build_model, train_model, evaluate_model_performance

def main():
    # Load in the data
    # Load in the data
    data_path = os.path.join("data", "admissions_data.csv")
    data = pd.read_csv(data_path)
    data = drop_cols(data, ['Serial No.'])
    print(data.head())

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, "Chance of Admit ")

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl")

    # Build and train the model
    model = build_model(X_train)
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=16)

    # Optionally, to visualize training history (requires matplotlib)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

    # Evaluate the model on test data and print results
    evaluate_model_performance(model, X_test, y_test)

    # Save the model
    model_path = os.path.join("models", "admissions_model")
    model.save(model_path)

if __name__ == "__main__":
    main()
