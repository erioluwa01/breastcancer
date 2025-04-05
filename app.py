import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('neural_network_model.h5')  # or .keras if you saved it in that format
scaler = pickle.load(open('scaling.pkl', 'rb'))  # Ensure this file exists in the same dir

# Homepage
@app.route('/')
def home():
    return render_template('home.html')

# API endpoint
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print("Received data:", data)

    # Preprocess input
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))

    # Predict
    output = model.predict(new_data)
    prediction_score = float(output[0][0])
    result = "Malignant" if prediction_score > 0.5 else "Benign"

    # Return JSON
    return jsonify({
        'prediction': prediction_score,
        'result': result
    })

if __name__ == "__main__":
    app.run(debug=True)
