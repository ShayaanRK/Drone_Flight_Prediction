import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Data generation
def generate_synthetic_data():
    np.random.seed(42)
    data = {
        "Battery Capacity (mAh)": np.random.randint(2000, 20001, 1000),
        "Battery Energy Density (Wh/kg)": np.random.uniform(100, 250, 1000),
        "Total Weight (kg)": np.random.uniform(1, 5, 1000),
        "Number of Motors": np.random.randint(4, 9, 1000),
        "Motor Power Consumption (W)": np.random.uniform(50, 500, 1000),
        "Propeller Efficiency (g/W)": np.random.uniform(5, 12, 1000),
        "Flight Speed (m/s)": np.random.uniform(3, 20, 1000),
    }
    df = pd.DataFrame(data)
    df["Flight Time (min)"] = (
        (df["Battery Capacity (mAh)"] * df["Battery Energy Density (Wh/kg)"]) /
        (df["Motor Power Consumption (W)"] * df["Total Weight (kg)"]) * 
        (df["Propeller Efficiency (g/W)"] * (df["Flight Speed (m/s)"] / 10))
    )
    return df

# Model training and evaluation
def train_models(df):
    X = df.drop(columns=["Flight Time (min)"])
    y = df["Flight Time (min)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "MAE": mae, "R2": r2}
        print(f"{name}: MSE={mse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

    return X_test, y_test, models["Random Forest"]

# Hyperparameter tuning
def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator

# Main function
def main():
    # Generate synthetic data
    df = generate_synthetic_data()

    # Train models
    X_test, y_test, best_model = train_models(df)

    # Hyperparameter tuning
    tuned_model = hyperparameter_tuning(X_test, y_test)

    # Save the best model
    joblib.dump(tuned_model, "models/flight_time_model.pkl")
    print("Model saved as 'models/flight_time_model.pkl'")

if __name__ == "__main__":
    main()
