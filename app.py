from flask import Flask, render_template, request, jsonify
from src.model_utils import predict_sentiment
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    sentiment = predict_sentiment(text)
    
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    # Ensure port is 5000 as it's standard for Flask
    app.run(debug=True, port=5000)
