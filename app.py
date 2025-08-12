# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file'].read()
    image = Image.open(io.BytesIO(file)).convert('L')  # Convert to grayscale
    image = image.resize((8, 8), Image.Resampling.LANCZOS)
    data = np.array(image, dtype=np.float64)
    data = (16 - (data / 16)).flatten()  # Scale to match dataset
    prediction = model.predict([data])[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
