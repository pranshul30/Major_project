# Algerian Forest Fire Prediction

This project predicts the **Fire Weather Index (FWI)** for two regions in Algeria: **Sidi-Bel** and **Brjaia**. The FWI is a crucial indicator used to estimate the likelihood of forest fires based on environmental parameters. The model has been developed using **Machine Learning (ML)** algorithms to assist in early fire detection and risk management.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model](#model)
- [Tech Stack](#tech-stack)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)


## Overview
The project provides a web-based interface where users can input various environmental factors such as temperature, humidity, wind speed, and rainfall to predict the Fire Weather Index. Based on the FWI value, the risk level of a forest fire is categorized from **Low** to **Extreme**.

## Features
- Predicts Fire Weather Index (FWI) using ML.
- Supports prediction for two regions: **Sidi-Bel** and **Brjaia**.
- Provides visual indicators for fire risk levels.
- User-friendly web interface for input and result display.

## Dataset
Kaggle-https://www.kaggle.com/code/sudhanshu432/algerian-forest-fires-prediction-regressions/notebook

The dataset used in this project contains historical environmental data including:
- Temperature (°C)
- Relative Humidity (%)
- Wind Speed (km/h)
- Rainfall (mm)
- Fine Fuel Moisture Code (FFMC)
- Duff Moisture Code (DMC)
- Initial Spread Index (ISI)
- Region Classification (0: Brjaia, 1: Sidi-Bel)

Source: Kaggle Notebook

Description:
The dataset contains 244 records of environmental data collected from:

Bejaia (Northeast Algeria) – 122 instances

Sidi Bel-Abbes (Northwest Algeria) – 122 instances

Period: June to September, 2012

Features:

Date: (DD/MM/YYYY)

Temperature (°C): 22 to 42

Relative Humidity (%): 21 to 90

Wind Speed (km/h): 6 to 29

Rainfall (mm): 0 to 16.8

FWI System Components:

FFMC (28.6 to 92.5)

DMC (1.1 to 65.9)

DC (7 to 220.4)

ISI (0 to 18.5)

BUI (1.1 to 68)

FWI (0 to 31.1)

Target Class:

Fire – 138 instances

Not Fire – 106 instances

## Model
The machine learning model has been trained using algorithms like:
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Elastic-Net**

The final model was selected based on evaluation metrics like **Mean Squared Error (MSE)** and **R-Squared Score**.

## Tech Stack
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask
- **Machine Learning:** Python, NumPy, Pandas, Scikit-Learn
- **Deployment:** Flask Web App

## Usage
- Enter the environmental parameters in the input form.
- Select the region (**Sidi-Bel** or **Brjaia**).
- Click on **Calculate FWI**.
- The model will display the predicted FWI and the corresponding risk level.


## Future Enhancements
- Integrate real-time weather data using external APIs.
- Implement additional regions.
- Deploy the application.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.



