from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)
CORS(app)

# ==================== CUSTOM FOCAL LOSS ====================
class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss', reduction='sum_over_batch_size'):
        super().__init__(name=name, reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        ce = -y_true * K.log(y_pred)
        weight = self.alpha * y_true * K.pow(1 - y_pred, self.gamma)
        focal_loss = weight * ce
        return K.sum(focal_loss, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ==================== MODEL LOADING ====================
# Try multiple model file names
MODEL_PATHS = ['best_emotion_optimized.h5']
model = None
MODEL_PATH = None

for path in MODEL_PATHS:
    if os.path.exists(path):
        try:
            print(f"Attempting to load model from: {path}")
            
            # Load model with custom objects
            model = keras.models.load_model(
                path,
                custom_objects={'FocalLoss': FocalLoss},
                compile=True
            )
            
            MODEL_PATH = path
            print("✓ Model loaded successfully")
            print(f"✓ Model path: {path}")
            print(f"✓ Model input shape: {model.input_shape}")
            print(f"✓ Model output shape: {model.output_shape}")
            break
            
        except Exception as e:
            print(f"✗ Failed to load {path}: {e}")
            continue

if model is None:
    print("⚠️  WARNING: No model could be loaded!")
    print("Available files in directory:")
    for file in os.listdir('.'):
        if file.endswith(('.h5', '.keras')):
            print(f"  - {file}")

# Emotion labels (must match training order)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

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
        'model_path': MODEL_PATH,
        'endpoints': {
            '/predict': 'POST - Predict emotion from base64 image',
            '/health': 'GET - Detailed health check'
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
            'error': 'Model not loaded. Please ensure model file exists.',
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
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'emotion': 'Neutral',
            'confidence': 0.0
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'emotions': EMOTION_LABELS,
        'input_shape': str(model.input_shape) if model else None,
        'output_shape': str(model.output_shape) if model else None
    })

if __name__ == '__main__':
    # Check if any model file exists
    found_models = [p for p in MODEL_PATHS if os.path.exists(p)]
    
    if not found_models:
        print("⚠️  Warning: No model files found!")
        print(f"Looking for: {', '.join(MODEL_PATHS)}")
        print("Please place your trained model (.h5 file) in the backend directory.")
    
    print("\n" + "="*60)
    print("🚀 EMOTION DETECTION API SERVER")
    print("="*60)
    print(f"Server running on: http://localhost:5000")
    print(f"Model loaded: {model is not None}")
    if MODEL_PATH:
        print(f"Model file: {MODEL_PATH}")
    print(f"Emotions: {', '.join(EMOTION_LABELS)}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)