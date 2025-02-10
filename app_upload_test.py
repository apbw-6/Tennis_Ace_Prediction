from flask import Flask, request, app, jsonify, render_template
import numpy as np
import pandas as pd
import dataframe_utils
import joblib
import os

app = Flask(__name__)

# Load the trained model (Assume it's saved as 'model.pkl')
loaded_model = joblib.load('model.joblib')


@app.route('/predict', methods=['POST'])
@app.route('/')
def home():
    return render_template("home_upload.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        test_data = pd.read_csv(file)
        required_features = ["speed", "angle", "spin", "height"]
        missing_features = [col for col in required_features if col not in test_data.columns]

        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        predictions = model.predict(test_data[required_features])
        test_data["prediction"] = predictions

        return test_data.to_json(orient="records")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
