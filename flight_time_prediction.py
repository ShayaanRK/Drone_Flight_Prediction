import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

    return X_train, X_test, y_train, y_test, results, models["Random Forest"]

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
    return grid_search.best_estimator_

# Predict flight time
def predict_flight_time(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

# Main function
def main():
    df = generate_synthetic_data()
    X_train, X_test, y_train, y_test, results, best_model = train_models(df)
    tuned_model = hyperparameter_tuning(X_train, y_train)

    # Predict flight time using the tuned model
    y_pred = predict_flight_time(tuned_model, X_test)

    # Display predictions
    predictions = pd.DataFrame({
        "Actual Flight Time (min)": y_test,
        "Predicted Flight Time (min)": y_pred
    })
    print("\nPredictions:")
    print(predictions.head(10))  # Display first 10 predictions

if __name__ == "__main__":
    main()