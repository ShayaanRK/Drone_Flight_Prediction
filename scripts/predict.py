import pandas as pd
import joblib

# Load the trained model
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Predict flight time
def predict_flight_time(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Main function
def main():
    # Load the trained model
    model = load_model("models/flight_time_model.pkl")

    # Define custom input
    custom_input = {
        "Battery Capacity (mAh)": 10000,
        "Battery Energy Density (Wh/kg)": 150,
        "Total Weight (kg)": 2.5,
        "Number of Motors": 6,
        "Motor Power Consumption (W)": 200,
        "Propeller Efficiency (g/W)": 8,
        "Flight Speed (m/s)": 10
    }
    custom_df = pd.DataFrame([custom_input])

    # Make prediction
    prediction = predict_flight_time(model, custom_df)
    print(f"Predicted Flight Time: {prediction[0]:.2f} minutes")

if __name__ == "__main__":
    main()
