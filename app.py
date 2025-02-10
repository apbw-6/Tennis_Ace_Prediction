import joblib
from flask import Flask, request, app, jsonify, render_template
import numpy as np
import pandas as pd
import dataframe_utils

app = Flask(__name__)

# Load model
model = joblib.load("model.joblib")


@app.route("/")  # Go to homepage
def home():
    return render_template("home.html")  # home.html yet to be created

    
@app.route('/predict', methods = ['POST'])
def predict():
    
    # Extract form data
    data = {
        'surface': request.form['surface'],
        'serve_side': request.form['serve_side'],
        'serve_number': request.form['serve_number'],
        'ball_hit_y': request.form['ball_hit_y'],
        'ball_hit_x': request.form['ball_hit_x'],
        'ball_hit_z': request.form['ball_hit_z'],
        'ball_hit_v': request.form['ball_hit_v'],
        'ball_net_v': request.form['ball_net_v'],
        'ball_net_z': request.form['ball_net_z'],
        'ball_net_y': request.form['ball_net_y'],
        'ball_bounce_x': request.form['ball_bounce_x'],
        'ball_bounce_y': request.form['ball_bounce_y'],
        'ball_bounce_v': request.form['ball_bounce_v'],
        'ball_bounce_angle': request.form['ball_bounce_angle'],
        'hitter_x': request.form['hitter_x'],
        'hitter_y': request.form['hitter_y'],
        'receiver_x': request.form['receiver_x'],
        'receiver_y': request.form['receiver_y'],
        'hitter_hand': request.form['hitter_hand'],
        'receiver_hand': request.form['receiver_hand']
    }
    
    # Convert data to a pandas DataFrame
    df = pd.DataFrame([data])
    
    # Convert numerical columns to appropriate data types
    for column in ['serve_number', 'ball_hit_y', 'ball_hit_x', \
       'ball_hit_z', 'ball_hit_v', 'ball_net_v', 'ball_net_z', 'ball_net_y', \
       'ball_bounce_x', 'ball_bounce_y', 'ball_bounce_v', 'ball_bounce_angle', \
       'hitter_x', 'hitter_y', 'receiver_x', 'receiver_y', 'hitter_hand', \
       'receiver_hand']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Clean data
    df = dataframe_utils.clean_dataframe(df)
    # In case there is some invalid data, df might be empty.
    if df.shape[0] == 0:
        return render_template("home.html", prediction_text= 'Invalid data. Please try again.')
    
    # Impute missing data
    df = dataframe_utils.impute_data(df)
    
    # Feature engineering
    df = dataframe_utils.feature_engineer(df)
    
    # Load model
    saved_model = joblib.load("xgb_model.joblib")
    # Predict
    pred = saved_model.predict(df)
    if pred[0] == 0:
        prediction = 'The serve is not an ace.'
    else:
        prediction = 'The serve is an ace!'
    
    return render_template("home.html", prediction_text= prediction)

if __name__ == "__main__":
    app.run(debug=True)