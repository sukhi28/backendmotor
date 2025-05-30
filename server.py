from flask import Flask, jsonify, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

def train_model():
    # Load the dataset
    df = pd.read_csv("data.csv")

    # Feature selection
    X = df[["Voltage", "Temperature", "DeltaWaterLevel", "MotorStatus"]]
    y = df["MotorHealth"]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train model on the entire dataset (since prediction will be on new data)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    return model, label_encoder

@app.route("/")
def home():
    return jsonify({"message": "Train and Predict API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data for a single prediction
        data = request.json

        # Validate input
        if not data or "sensor_data" not in data:
            return jsonify({"error": "Missing 'sensor_data'"}), 400

        # Convert input into array and reshape for a single sample
        sensor_data = np.array(data["sensor_data"]).reshape(1, -1)

        # Ensure input has the correct number of features
        if sensor_data.shape[1] != 4:
            return jsonify({
                "error": "Each sensor data row must have exactly 4 values: [Voltage, Temperature, DeltaWaterLevel, MotorStatus]"
            }), 400

        # Train the model
        model, label_encoder = train_model()

        # Make prediction
        prediction = model.predict(sensor_data)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()