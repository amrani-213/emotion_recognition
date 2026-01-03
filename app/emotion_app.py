# ============================================================================
# STREAMLIT REAL-TIME EMOTION DETECTION APP
# Save as: emotion_app.py
# Run with: streamlit run emotion_app.py
# ============================================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time
from collections import deque
import pandas as pd

# ============================================================================
# MODEL ARCHITECTURE (Must match your trained model)
# ============================================================================

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class ResNetWithCBAM(nn.Module):
    """ResNet18 + CBAM for Emotion Recognition"""
    
    def __init__(self, num_classes=7, dropout_rate=0.25):
        super(ResNetWithCBAM, self).__init__()
        
        resnet = models.resnet18(weights=None)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.cbam1 = CBAM(64, reduction_ratio=16)
        self.cbam2 = CBAM(128, reduction_ratio=16)
        self.cbam3 = CBAM(256, reduction_ratio=16)
        self.cbam4 = CBAM(512, reduction_ratio=16)
        
        self.avgpool = resnet.avgpool
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.cbam1(x)
        
        x = self.layer2(x)
        x = self.cbam2(x)
        
        x = self.layer3(x)
        x = self.cbam3(x)
        
        x = self.layer4(x)
        x = self.cbam4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model(model_path='\models\emotion_resnet18_cbam_latest.pth'):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = ResNetWithCBAM(num_classes=7, dropout_rate=0.25)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    emotion_classes = checkpoint['emotion_classes']
    
    return model, emotion_classes, device


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_frame(frame):
    """
    Preprocess frame for model input
    CRITICAL: FER2013 is grayscale dataset, so we convert to gray first
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # FER2013-style preprocessing: BGR → GRAY → RGB (3-channel)
    # This matches training data format (grayscale converted to 3-channel)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Apply transforms
    tensor = transform(pil_image)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    return tensor


# ============================================================================
# FACE DETECTION - MediaPipe (Better than Haar Cascade)
# ============================================================================

@st.cache_resource
def load_face_detector():
    """Load MediaPipe face detector (better than Haar Cascade)"""
    try:
        mp_face_detection = mp.solutions.face_detection
        return mp_face_detection.FaceDetection(
            model_selection=0,  # 0 = short-range (webcam), 1 = full-range
            min_detection_confidence=0.5
        )
    except AttributeError:
        # Fallback to Haar Cascade if MediaPipe not properly installed
        st.warning("⚠️ MediaPipe not available, using Haar Cascade fallback")
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        return face_cascade


def detect_faces(frame, face_detector):
    """
    Detect faces using MediaPipe (or Haar Cascade as fallback)
    Returns list of (x, y, w, h) face boxes
    """
    # Check if MediaPipe or Haar Cascade
    if hasattr(face_detector, 'process'):
        # MediaPipe detection
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = face_detector.process(frame_rgb)
        
        faces = []
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                
                # Convert normalized coordinates to pixel coordinates
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                
                # Clip to image bounds
                x = max(0, x)
                y = max(0, y)
                bw = min(bw, w - x)
                bh = min(bh, h - y)
                
                faces.append((x, y, bw, bh))
    else:
        # Haar Cascade fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
    
    return faces


# ============================================================================
# EMOTION PREDICTION
# ============================================================================

def predict_emotion(face_img, model, device):
    """Predict emotion from face image"""
    tensor = preprocess_frame(face_img)
    tensor = tensor.to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    
    return predicted.item(), confidence.item(), probabilities.cpu().numpy()


# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_results(frame, faces, emotion_classes, predictions):
    """Draw bounding boxes and emotion labels"""
    
    # Emotion colors (BGR format)
    emotion_colors = {
        'angry': (0, 0, 255),      # Red
        'disgust': (0, 255, 255),  # Yellow
        'fear': (128, 0, 128),     # Purple
        'happy': (0, 255, 0),      # Green
        'neutral': (255, 255, 255), # White
        'sad': (255, 0, 0),        # Blue
        'surprise': (0, 165, 255)  # Orange
    }
    
    for (x, y, w, h), (pred_idx, conf, probs) in zip(faces, predictions):
        emotion = emotion_classes[pred_idx]
        color = emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Draw emotion label with background
        label = f"{emotion.upper()}: {conf*100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Background rectangle
        cv2.rectangle(frame, 
                     (x, y - text_height - 10), 
                     (x + text_width + 10, y), 
                     color, -1)
        
        # Text
        cv2.putText(frame, label, (x + 5, y - 5), 
                   font, font_scale, (0, 0, 0), thickness)
    
    return frame


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Real-Time Emotion Detection",
        page_icon="😊",
        layout="wide"
    )
    
    # Title
    st.title("🎭 Real-Time Emotion Detection")
    st.markdown("**Powered by ResNet18 + CBAM Attention** | 69% Accuracy | MediaPipe Face Detection")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path",
        "models/emotion_resnet18_cbam_latest.pth"
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Show probabilities
    show_probabilities = st.sidebar.checkbox("Show All Probabilities", value=True)
    
    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode",
        ["📷 Webcam", "🖼️ Upload Image"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Emotion Classes")
    st.sidebar.markdown("""
    - 😠 Angry
    - 🤢 Disgust
    - 😨 Fear
    - 😊 Happy
    - 😐 Neutral
    - 😢 Sad
    - 😲 Surprise
    """)
    
    # Load model
    try:
        model, emotion_classes, device = load_model(model_path)
        face_detector = load_face_detector()  # ✅ Renamed from face_cascade
        st.sidebar.success("✅ Model loaded successfully!")
        st.sidebar.info("🔍 Using MediaPipe face detection")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    
    # ========================================================================
    # WEBCAM MODE
    # ========================================================================
    
    if mode == "📷 Webcam":
        st.markdown("### Live Webcam Feed")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            run_webcam = st.checkbox("Start Webcam", value=False)
            FRAME_WINDOW = st.image([])
        
        with col2:
            st.markdown("### 📈 Live Statistics")
            emotion_placeholder = st.empty()
            confidence_placeholder = st.empty()
            prob_placeholder = st.empty()
        
        if run_webcam:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Cannot access webcam!")
                st.stop()
            
            fps_time = time.time()
            fps = 0
            
            # ✅ Emotion smoothing buffer (reduces flicker)
            emotion_buffer = deque(maxlen=5)  # Average last 5 predictions
            
            while run_webcam:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Failed to capture frame")
                    break
                
                # Detect faces
                faces = detect_faces(frame, face_detector)  # ✅ Using face_detector
                
                # Predict emotions for each face
                predictions = []
                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    
                    if face_img.size > 0:
                        pred_idx, conf, probs = predict_emotion(face_img, model, device)
                        
                        # ✅ Apply emotion smoothing for first face only
                        if len(predictions) == 0:  # First face
                            emotion_buffer.append(probs)
                            
                            if len(emotion_buffer) > 0:
                                # Average probabilities over buffer (reduces flicker)
                                avg_probs = np.mean(emotion_buffer, axis=0)
                                pred_idx = np.argmax(avg_probs)
                                conf = avg_probs[pred_idx]
                                probs = avg_probs  # Use smoothed probabilities
                        
                        # ✅ Apply confidence threshold (only show if above threshold)
                        if conf >= confidence_threshold:
                            predictions.append((pred_idx, conf, probs))
                
                # Draw results
                if len(faces) > 0 and len(predictions) > 0:
                    frame = draw_results(frame, faces, emotion_classes, predictions)
                    
                    # Update statistics for first face
                    pred_idx, conf, probs = predictions[0]
                    emotion = emotion_classes[pred_idx]
                    
                    with col2:
                        emotion_placeholder.markdown(f"### 🎭 Detected: **{emotion.upper()}**")
                        confidence_placeholder.metric("Confidence", f"{conf*100:.1f}%")
                        
                        if show_probabilities:
                            prob_df = pd.DataFrame({
                                'Emotion': [e.capitalize() for e in emotion_classes],
                                'Probability': probs * 100
                            }).sort_values('Probability', ascending=False)
                            
                            prob_placeholder.bar_chart(
                                prob_df.set_index('Emotion')['Probability']
                            )
                
                # Calculate FPS
                fps = 1 / (time.time() - fps_time)
                fps_time = time.time()
                
                # Display FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb, channels="RGB", use_container_width=True)
            
            cap.release()
    
    # ========================================================================
    # IMAGE UPLOAD MODE
    # ========================================================================
    
    else:
        st.markdown("### Upload an Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Original Image")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                        use_container_width=True)
            
            # Detect faces
            faces = detect_faces(frame, face_detector)  # ✅ Using face_detector
            
            if len(faces) == 0:
                st.warning("⚠️ No faces detected in the image!")
            else:
                # Predict emotions
                predictions = []
                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    
                    if face_img.size > 0:
                        pred_idx, conf, probs = predict_emotion(face_img, model, device)
                        predictions.append((pred_idx, conf, probs))
                
                # Draw results
                result_frame = draw_results(frame.copy(), faces, emotion_classes, predictions)
                
                with col2:
                    st.markdown("#### Detected Emotions")
                    st.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), 
                            use_container_width=True)
                
                # Show detailed results
                st.markdown("### 📊 Detailed Results")
                
                for i, ((x, y, w, h), (pred_idx, conf, probs)) in enumerate(zip(faces, predictions)):
                    emotion = emotion_classes[pred_idx]
                    
                    st.markdown(f"#### Face {i+1}")
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric("Emotion", emotion.upper())
                        st.metric("Confidence", f"{conf*100:.1f}%")
                    
                    with col_b:
                        if show_probabilities:
                            prob_df = pd.DataFrame({
                                'Emotion': [e.capitalize() for e in emotion_classes],
                                'Probability (%)': probs * 100
                            }).sort_values('Probability (%)', ascending=False)
                            
                            st.dataframe(prob_df, use_container_width=True)


if __name__ == "__main__":
    import pandas as pd
    main()