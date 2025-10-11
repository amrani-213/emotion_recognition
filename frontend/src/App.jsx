import React from 'react'
import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './App.css'

const EMOTION_EMOJIS = {
  Angry: '😠',
  Disgust: '🤢',
  Fear: '😨',
  Happy: '😄',
  Sad: '😢',
  Surprise: '😮',
  Neutral: '😐',
}

const API_URL = 'http://localhost:5000/predict'
const CAPTURE_INTERVAL = 1500 // 1.5 seconds

function App() {
  const [emotion, setEmotion] = useState('Neutral')
  const [confidence, setConfidence] = useState(0)
  const [isWebcamActive, setIsWebcamActive] = useState(false)
  const [error, setError] = useState(null)
  const [isPulsing, setIsPulsing] = useState(false)

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const intervalRef = useRef(null)

  // Initialize webcam
  useEffect(() => {
    const startWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        })

        if (videoRef.current) {
          videoRef.current.srcObject = stream
          setIsWebcamActive(true)
          setError(null)
        }
      } catch (err) {
        console.error('Error accessing webcam:', err)
        setError('Unable to access webcam. Please grant camera permissions.')
      }
    }

    startWebcam()

    // Cleanup: stop webcam on unmount
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks()
        tracks.forEach((track) => track.stop())
      }
    }
  }, [])

  // Start capturing and sending frames
  useEffect(() => {
    if (!isWebcamActive) return

    const captureAndPredict = async () => {
      try {
        const canvas = canvasRef.current
        const video = videoRef.current

        if (!canvas || !video || video.readyState !== 4) return

        const context = canvas.getContext('2d')
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight

        // Draw current video frame to canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height)

        // Convert canvas to base64 image
        const base64Image = canvas.toDataURL('image/jpeg', 0.8)

        // Send to backend for prediction
        const response = await axios.post(
          API_URL,
          {
            image: base64Image,
          },
          {
            timeout: 5000,
          }
        )

        const { emotion: predictedEmotion, confidence: conf } = response.data

        // Trigger pulse animation if emotion changed
        if (predictedEmotion !== emotion) {
          setIsPulsing(true)
          setTimeout(() => setIsPulsing(false), 600)
        }

        setEmotion(predictedEmotion)
        setConfidence(conf)
        setError(null)
      } catch (err) {
        console.error('Prediction error:', err)
        if (err.code === 'ECONNABORTED') {
          setError('Request timeout. Backend may be slow.')
        } else if (err.response) {
          setError(`Server error: ${err.response.status}`)
        } else if (err.request) {
          setError('Backend not responding. Is Flask running?')
        } else {
          setError('Failed to predict emotion')
        }
      }
    }

    // Start interval for continuous prediction
    intervalRef.current = setInterval(captureAndPredict, CAPTURE_INTERVAL)

    // Cleanup interval on unmount
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [isWebcamActive, emotion])

  return (
    <div className="app">
      <header className="header">
        <h1>🎭 Real-Time Emotion Detection</h1>
        <p className="subtitle">
          AI-powered facial emotion recognition using deep learning
        </p>
      </header>

      <main className="main">
        {error && <div className="error-banner">⚠️ {error}</div>}

        <div className="webcam-container">
          <video ref={videoRef} autoPlay playsInline muted className="webcam" />

          {!isWebcamActive && (
            <div className="webcam-loading">
              <div className="spinner"></div>
              <p>Initializing webcam...</p>
            </div>
          )}

          <div className={`emotion-overlay ${isPulsing ? 'pulse' : ''}`}>
            <div className="emoji">{EMOTION_EMOJIS[emotion]}</div>
            <div className="emotion-text">{emotion}</div>
            <div className="confidence">
              {(confidence * 100).toFixed(1)}% confident
            </div>
          </div>
        </div>

        {/* Hidden canvas for frame capture */}
        <canvas ref={canvasRef} style={{ display: 'none' }} />

        <div className="info-card">
          <h3>How it works</h3>
          <ol>
            <li>Your webcam captures your face in real-time</li>
            <li>Every 1.5 seconds, a frame is sent to the AI model</li>
            <li>The model predicts your current emotion</li>
            <li>The emoji updates to reflect your mood</li>
          </ol>
        </div>
      </main>

      <footer className="footer">
        <p>Built with React, Flask & TensorFlow | Trained on FER2013 Dataset</p>
      </footer>
    </div>
  )
}

export default App
