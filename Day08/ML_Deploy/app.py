from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "House Price Prediction API is running!"

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)

        # Expecting a list of feature values
        features = np.array(data["features"]).reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)

        # Return prediction
        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
