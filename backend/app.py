from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load the trained emotion detection model
MODEL_PATH = 'best_emotion_model.h5'

try:
    model = keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully")
    print(f"✓ Model input shape: {model.input_shape}")
    print(f"✓ Model output shape: {model.output_shape}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# Emotion labels (must match training order)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(image_data):
    """
    Preprocess base64 image for model prediction
    - Decode base64 to image
    - Convert to grayscale
    - Resize to 48x48
    - Normalize to [0, 1]
    - Reshape to (1, 48, 48, 1)
    """
    try:
        # Remove base64 header if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to 48x48
        resized = cv2.resize(gray, (48, 48))
        
        # Normalize pixel values to [0, 1]
        normalized = resized / 255.0
        
        # Reshape to (1, 48, 48, 1) for model input
        preprocessed = normalized.reshape(1, 48, 48, 1)
        
        return preprocessed
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Emotion Detection API is running',
        'model_loaded': model is not None,
        'endpoints': {
            '/predict': 'POST - Predict emotion from base64 image'
        }
    })

@app.route('/predict', methods=['POST'])
def predict_emotion():
    """
    Predict emotion from base64-encoded image
    Expected JSON: { "image": "base64_string" }
    Returns: { "emotion": "Happy", "confidence": 0.95, "probabilities": {...} }
    """
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'emotion': 'Neutral',
            'confidence': 0.0
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'error': 'No image data provided',
                'emotion': 'Neutral',
                'confidence': 0.0
            }), 400
        
        # Preprocess the image
        image_data = data['image']
        preprocessed_image = preprocess_image(image_data)
        
        if preprocessed_image is None:
            return jsonify({
                'error': 'Failed to preprocess image',
                'emotion': 'Neutral',
                'confidence': 0.0
            }), 400
        
        # Make prediction
        predictions = model.predict(preprocessed_image, verbose=0)
        
        # Get predicted emotion
        predicted_index = np.argmax(predictions[0])
        predicted_emotion = EMOTION_LABELS[predicted_index]
        confidence = float(predictions[0][predicted_index])
        
        # Create probability dictionary for all emotions
        probabilities = {
            EMOTION_LABELS[i]: float(predictions[0][i]) 
            for i in range(len(EMOTION_LABELS))
        }
        
        # Log prediction
        print(f"Predicted: {predicted_emotion} ({confidence:.2%})")
        
        return jsonify({
            'emotion': predicted_emotion,
            'confidence': confidence,
            'probabilities': probabilities
        })
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({
            'error': str(e),
            'emotion': 'Neutral',
            'confidence': 0.0
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'emotions': EMOTION_LABELS
    })

if __name__ == '__main__':
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Warning: Model file '{MODEL_PATH}' not found!")
        print("Please place 'emotion_model.h5' in the backend directory.")
    
    print("\n" + "="*60)
    print("🚀 EMOTION DETECTION API SERVER")
    print("="*60)
    print(f"Server running on: http://localhost:5000")
    print(f"Model loaded: {model is not None}")
    print(f"Emotions: {', '.join(EMOTION_LABELS)}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)