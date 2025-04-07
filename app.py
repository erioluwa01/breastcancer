import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# Load the trained model and scaler
model = load_model('neural_network_model.keras')
scaler = pickle.load(open('scaling.pkl', 'rb'))

app = Flask(__name__)

# Route for the homepage that displays the form
@app.route('/')
def home():
    return render_template('home.html')

# Route to handle predictions
@app.route('/predict_api', methods=['POST'])
def predict_api():
    # List of features corresponding to the form fields
    fields = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
        'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ]
    
    # Collecting the form data
    data = [float(request.form[field]) for field in fields]
    
    # Scaling the input data using the pre-fitted scaler
    new_data = scaler.transform(np.array(data).reshape(1, -1))
    
    # Predicting with the model
    output = model.predict(new_data)
    
    # Prediction score and result classification
    prediction_score = float(output[0][0])
    result = "Malignant" if prediction_score > 0.5 else "Benign"

    # Returning the prediction in JSON format
    return jsonify({
        'prediction': prediction_score,
        'result': result
    })

if __name__ == "__main__":
    app.run(debug=True)
