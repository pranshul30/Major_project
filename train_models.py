import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import json

# Load data
df = pd.read_csv("Algerian_forest_fires_cleaned_dataset.csv")

# Features and target
X = df.drop(columns=["Classes", "Region"])
y = df["FWI"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models to train
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# Store results
model_results = {}

# Train each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Save model
    with open(f"MODELS/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save results
    model_results[name] = {
        "RMSE": rmse,
        "R2": r2
    }

# Save results to JSON
with open("MODELS/results.json", "w") as f:
    json.dump(model_results, f, indent=4)

print("All models trained, saved, and results stored.")
