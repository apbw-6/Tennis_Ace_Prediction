import joblib
from flask import Flask, request, app, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("model.joblib")


@app.route("/")  # Go to homepage
def home():
    return render_template("home.html")  # home.html yet to be created


@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]

    # We want to treat data as a dataframe and copy all actions that were done on test data set.
    df = pd.DataFrame([data])
    
    df["households"] = np.log(df["households"] + 1)
    df["population"] = np.log(df["population"] + 1)
    df["total_bedrooms"] = np.log(df["total_bedrooms"] + 1)
    df["total_rooms"] = np.log(df["total_rooms"] + 1)

    # One-hot-encoding
    df["<1H OCEAN"] = 0
    df["INLAND"] = 0
    df["ISLAND"] = 0
    df["NEAR BAY"] = 0
    df["NEAR OCEAN"] = 0
    df[df["ocean_proximity"][0]] = 1
    df = df.drop(columns=["ocean_proximity"])

    df["fraction_of_bedrooms"] = df["total_bedrooms"] / df["total_rooms"]
    df["rooms_per_household"] = df["total_rooms"] / df["households"]

    prediction = model.predict(df)
    # model.predict() returns array, and we need the first element
    print("Prediction: ", prediction[0])

    return jsonify(prediction[0])
    
@app.route('/predict', methods = ['POST'])
def predict():
    # Extract form data
    data = {
        'longitude': request.form['longitude'],
        'latitude': request.form['latitude'],
        'housing_median_age': request.form['housing_median_age'],
        'total_rooms': request.form['total_rooms'],
        'total_bedrooms': request.form['total_bedrooms'],
        'population': request.form['population'],
        'households': request.form['households'],
        'median_income': request.form['median_income'],
        'ocean_proximity': request.form['ocean_proximity']
    }
    
    # Convert data to a pandas DataFrame
    df = pd.DataFrame([data])
    # Convert numerical columns to appropriate data types
    for column in ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                   'total_bedrooms', 'population', 'households', 'median_income']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    df["households"] = np.log(df["households"] + 1)
    df["population"] = np.log(df["population"] + 1)
    df["total_bedrooms"] = np.log(df["total_bedrooms"] + 1)
    df["total_rooms"] = np.log(df["total_rooms"] + 1)

    # One-hot-encoding
    df["<1H OCEAN"] = 0
    df["INLAND"] = 0
    df["ISLAND"] = 0
    df["NEAR BAY"] = 0
    df["NEAR OCEAN"] = 0
    df[df["ocean_proximity"][0]] = 1
    df = df.drop(columns=["ocean_proximity"])

    df["fraction_of_bedrooms"] = df["total_bedrooms"] / df["total_rooms"]
    df["rooms_per_household"] = df["total_rooms"] / df["households"]

    prediction = model.predict(df)
    
    return render_template("home.html", prediction_text= 'The predicted house price is ${:,.2f}.'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)