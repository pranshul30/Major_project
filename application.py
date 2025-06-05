import os
import numpy as np
import pandas as pd
import pickle
import warnings
from flask import Flask, request, render_template ,redirect,url_for

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --------- MODEL 1: Logistic Regression for Forest Fire Danger ---------
# Check if model exists
if not os.path.exists("model1.pkl"):
    data = pd.read_csv("Forest_fire.csv")
    data = np.array(data)

    # Extract features and target
    X = data[1:, 1:-1]   # Skip header row, take columns 1 to second-last
    y = data[1:, -1]     # Last column is 'Classes'

    # Convert to integers
    X = X.astype('int')
    y = y.astype('int')

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Train logistic regression model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Save model
    pickle.dump(log_reg, open('model1.pkl', 'wb'))

model1 = pickle.load(open('model1.pkl', 'rb'))  # Forest fire classifier

# --------- MODEL 2: Linear Regression for FWI prediction ---------
model2 = pickle.load(open("MODELS/model.pkl", "rb"))  # Linear regressor
scaler = pickle.load(open("MODELS/Scaler.pkl", "rb"))

# --------- ROUTES ---------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict_fire_danger', methods=['POST'])
def predict_fire_danger():
    int_features = [float(x) for x in request.form.values()]  # Changed int -> float for more flexibility
    final = [np.array(int_features)]
    prediction = model1.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    msg = f"🔥 Your Forest is in Danger. Probability of fire occurring is {output}" if float(output) > 0.5 \
        else f"✅ Your Forest is Safe. Probability of fire occurring is {output}"
    return render_template('forest_fire.html', pred=msg)

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'POST':
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))

        new_data = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI]])
        results = model2.predict(new_data)

        return render_template('result.html', results=results[0])
    else:
        return render_template("home.html")

# Informational Pages
@app.route('/weather-fires')
def weather_fires():
    return render_template('weather-fires.html')

@app.route('/fireindices')
def fire_indices():
    return render_template('fireindices.html')

@app.route('/temperature-impact')
def temperature_impact():
    return render_template('temperature-impact.html')

@app.route("/predict_fire_danger")
def custom_model():
    return render_template("forest_fire.html")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predictdata')
def fp():
    return render_template("home.html")

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html'), 404


if __name__ == '__main__':
    app.run(debug=True)

