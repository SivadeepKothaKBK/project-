import streamlit as st
import joblib
import cv2
import numpy as np

# Feature extraction from videos
def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to reduce computation
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        frame_count += 1
        
        # Extract every 10th frame
        if frame_count % 10 == 0:
            break
    
    cap.release()
    
    # Flatten frames and take mean to create feature vector
    frames = np.array(frames)
    feature_vector = frames.mean(axis=0).flatten()
    return feature_vector

# Streamlit Web App
st.markdown("<h1 style='font-size:48px;'>"
            "<span style='color:#4285F4;'>C</span>"
            "<span style='color:#EA4335;'>r</span>"
            "<span style='color:#FBBC05;'>e</span>"
            "<span style='color:#4285F4;'>d</span>"
            "<span style='color:#34A853;'>i</span>"
            "<span style='color:#EA4335;'>t</span> "
            "<span style='color:#FBBC05;'>C</span>"
            "<span style='color:#34A853;'>a</span>"
            "<span style='color:#4285F4;'>r</span>"
            "<span style='color:#EA4335;'>d</span> "
            "<span style='color:#34A853;'>F</span>"
            "<span style='color:#FBBC05;'>r</span>"
            "<span style='color:#4285F4;'>a</span>"
            "<span style='color:#EA4335;'>u</span>"
            "<span style='color:#34A853;'>d</span> "
            "<span style='color:#FBBC05;'>D</span>"
            "<span style='color:#4285F4;'>e</span>"
            "<span style='color:#34A853;'>t</span>"
            "<span style='color:#EA4335;'>e</span>"
            "<span style='color:#FBBC05;'>c</span>"
            "<span style='color:#4285F4;'>t</span>"
            "<span style='color:#34A853;'>i</span>"
            "<span style='color:#EA4335;'>o</span>"
            "<span style='color:#FBBC05;'>n</span> "
            "<span style='color:#34A853;'>i</span>"
            "<span style='color:#4285F4;'>n</span> "
            "<span style='color:#EA4335;'>V</span>"
            "<span style='color:#34A853;'>i</span>"
            "<span style='color:#FBBC05;'>d</span>"
            "<span style='color:#4285F4;'>e</span>"
            "<span style='color:#EA4335;'>o</span>"
            "<span style='color:#34A853;'>s</span>"
            "</h1>", unsafe_allow_html=True)

# Load model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Video file upload
uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    # Display the video
    st.video("temp_video.mp4")
    
    # Button to predict fraud
    if st.button("Predict Fraud"):
        # Extract features from the uploaded video
        features = extract_features("temp_video.mp4")
        features_scaled = scaler.transform([features])
        
        # Predict fraud
        prediction = model.predict(features_scaled)
        
        # Display result
        if prediction[0] == 1:
            st.markdown("### The video contains fraud activity.", unsafe_allow_html=True)
            st.markdown("<span style='color:red; font-size:24px;'>Fraud Detected</span>", unsafe_allow_html=True)
        else:
            st.markdown("### The video does not contain fraud activity.", unsafe_allow_html=True)
            st.markdown("<span style='color:green; font-size:24px;'>No Fraud Detected</span>", unsafe_allow_html=True)
