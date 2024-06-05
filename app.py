from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Assuming your model loading code is here

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])  # Allow both GET and POST requests
def predict():
    if request.method == 'POST':
        data = request.get_json()
        features = data.get('features')  # Extract features from JSON data
        # Perform prediction and return result
        return jsonify({'prediction': prediction_result})
    else:
        return 'Only POST requests are allowed for this endpoint', 405

if __name__ == '__main__':
    app.run(debug=True)
