# Retraining with 7 features only
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("Algerian_forest_fires_cleaned_dataset.csv")

# retrain_linear_model.py

# Load data


# Select only the 7 features
X = data[['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI']]
y = data['FWI']  # Assuming you are predicting FWI (Fire Weather Index)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LinearRegression()
model.fit(X_scaled, y)

# Save model and scaler
with open("MODELS/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("MODELS/Scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
