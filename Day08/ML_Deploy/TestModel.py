import joblib

# Load model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example new data (1 row)
new_data = [[8.3252, 41, 6.9841, 1.0238, 322, 2.5556, 37.88, -122.23]]
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)
print("Predicted House Value:", prediction[0])
