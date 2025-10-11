# 🎭 Real-Time Emotion Detection AI

Full-stack web application for real-time facial emotion recognition using deep learning. Detects 7 emotions through webcam with a CNN model trained on FER2013 dataset.

---

## 🌟 Features

- Real-time emotion detection from webcam (1.5s intervals)
- 7 emotion classes: Angry 😠, Disgust 🤢, Fear 😨, Happy 😄, Sad 😢, Surprise 😮, Neutral 😐
- CNN model trained on 35,000+ FER2013 images
- Confidence scores for predictions
- Animated emoji overlay with pulse effects
- REST API with CORS support

---

## 🛠️ Tech Stack

**Frontend:** React 18, Vite, Axios, CSS3  
**Backend:** Flask, TensorFlow 2.15, Keras, OpenCV, Flask-CORS  
**ML:** CNN (4 conv blocks, BatchNorm, Dropout), 8M parameters  
**Dataset:** FER2013 (48x48 grayscale images)

---

## 📁 Project Structure

```
emotion-detection/
├── backend/
│   ├── app.py
│   ├── best_emotion_model.h5
│   ├── requirements.txt
│   └── .gitignore
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── main.jsx
│   ├── index.css
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   └── .gitignore
├── .gitignore
└── README.md
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11
- Node.js 16+
- Webcam

### Backend Setup

```bash
cd backend

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Place best_emotion_model.h5 in backend/ directory

python app.py
```

Backend runs on `http://localhost:5000`

### Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

Frontend runs on `http://localhost:5173`

### Usage

Open `http://localhost:5173` and allow webcam access when prompted.

---

## 🔧 API Endpoints

**Base URL:** `http://localhost:5000`

### POST /predict

**Request:**

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response:**

```json
{
  "emotion": "Happy",
  "confidence": 0.8523,
  "probabilities": {
    "Angry": 0.0234,
    "Disgust": 0.0012,
    "Fear": 0.0098,
    "Happy": 0.8523,
    "Sad": 0.0456,
    "Surprise": 0.0521,
    "Neutral": 0.0156
  }
}
```

### GET /

Health check endpoint

### GET /health

Detailed system status

---

## 🧠 Model Details

**Architecture:**

- Input: 48x48 grayscale images
- 4 Convolutional blocks with BatchNorm, MaxPooling, Dropout
- 2 Dense layers (512 → 256)
- Softmax output (7 classes)
- Total parameters: ~8 million

**Training:**

- Dataset: FER2013 (35,887 images)
- Split: 80% train, 10% val, 10% test
- Augmentation: Rotation, shift, flip, zoom
- Optimizer: Adam (lr=0.0001)
- Loss: Categorical Crossentropy
- Hardware: Kaggle GPU T4
- Training time: 30-40 minutes
- Test accuracy: 60-68%
- Inference time: 50-100ms per image

---

## 🐛 Troubleshooting

| Problem                | Solution                                              |
| ---------------------- | ----------------------------------------------------- |
| Model not found        | Place `best_emotion_model.h5` in `backend/` directory |
| CORS error             | Install `flask-cors`: `pip install flask-cors`        |
| Port 5000 in use       | Change port in `app.py`: `app.run(port=5001)`         |
| Webcam not working     | Check browser permissions, use Chrome                 |
| Backend not responding | Verify Flask is running, check `API_URL` in `App.jsx` |

---

## 🎨 Customization

### Change capture interval

In `frontend/src/App.jsx`:

```javascript
const CAPTURE_INTERVAL = 2000 // Change to 2 seconds
```

### Change UI colors

In `frontend/src/App.css`:

```css
body {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.webcam-container {
  border: 4px solid #YOUR_COLOR;
}
```

### Add new emotions

1. Retrain model with new classes
2. Update `EMOTION_LABELS` in `backend/app.py`
3. Add emoji mapping in `frontend/src/App.jsx`

---

## 📊 FER2013 Dataset

- Size: 35,887 images (48x48 grayscale)
- Classes distribution:
  - Happy: 8,989
  - Neutral: 6,198
  - Sad: 6,077
  - Fear: 5,121
  - Angry: 4,953
  - Surprise: 4,002
  - Disgust: 547 (imbalanced)

---

## 📝 License

MIT License

---

## 👨‍💻 Contact

**Kaggle:** [@serhiikravchenko2009](https://www.kaggle.com/serhiikravchenko2009)
**GitHub:** [@Serhii2009](https://github.com/Serhii2009)
**LinkedIn:** [serhii-kravchenko-b941272a6](https://www.linkedin.com/in/serhii-kravchenko-b941272a6/)

---

## 📚 Resources

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [React Documentation](https://react.dev)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**⭐ If you found this project helpful, please consider giving it a star!**
**Built with React, Flask & TensorFlow**
