import io
import re
import base64
import numpy as np

from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps

import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = 'models/mnist_cnn.h5'

# LOAD MODEL
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")

except Exception as e:
    model = None
    print(f"ERROR loading model: {e}")

CONFIDENCE_THRESHOLD = 0.70


# HOME PAGE
@app.route('/')
def home():
    return render_template('index.html')


# IMAGE PREPROCESSING
def preprocess_image_from_base64(b64data):

    match = re.search(r'base64,(.*)', b64data)

    if not match:
        raise ValueError('Invalid image data')

    img_bytes = base64.b64decode(match.group(1))

    # IMPORTANT -> HANDLE TRANSPARENCY
    img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')

    white_bg = Image.new('RGBA', img.size, 'WHITE')
    img = Image.alpha_composite(white_bg, img)

    # CONVERT TO GRAYSCALE
    img = img.convert('L')

    # INVERT -> BLACK BG / WHITE DIGIT
    img = ImageOps.invert(img)

    arr = np.array(img)

    # REMOVE NOISE
    coords = np.column_stack(np.where(arr > 30))

    if len(coords) == 0:
        raise ValueError('Canvas is empty')

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    arr = arr[y_min:y_max + 1, x_min:x_max + 1]

    img = Image.fromarray(arr)

    # SCALE DIGIT
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # CENTER ON 28x28
    canvas = Image.new('L', (28, 28), color=0)

    paste_x = (28 - img.width) // 2
    paste_y = (28 - img.height) // 2

    canvas.paste(img, (paste_x, paste_y))

    arr = np.array(canvas).astype('float32') / 255.0

    # CNN INPUT SHAPE
    arr = np.expand_dims(arr, axis=(0, -1))

    return arr


# PREDICTION ROUTE
@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500

    try:

        data = request.get_json()

        if 'image' not in data:
            return jsonify({
                'error': 'No image received'
            }), 400

        x = preprocess_image_from_base64(data['image'])

        probs = model.predict(x, verbose=0)[0]

        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])

        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence * 100, 1),
            'probabilities': [
                round(float(p) * 100, 1)
                for p in probs
            ],
            'low_confidence': confidence < CONFIDENCE_THRESHOLD
        })

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 500


# START SERVER
if __name__ == '__main__':
    app.run(debug=True)