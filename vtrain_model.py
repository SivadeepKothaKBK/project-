import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Path to video files
video_path = r"C:\Users\sivad\OneDrive\Desktop\project\videos\videos"

# Video labels
target_labels = {
    "3.mp4": 1, "5.mp4": 1, "10.mp4": 1, "11.mp4": 1, "12.mp4": 0,
    "13.mp4": 1, "14.mp4": 0, "15.mp4": 0, "16.mp4": 1, "17.mp4": 1,
    "18.mp4": 1, "19.mp4": 0, "20.mp4": 0, "21.mp4": 0, "22.mp4": 0,
    "23.mp4": 0, "24.mp4": 0, "25.mp4": 0, "27.mp4": 1, "28.mp4": 1,
    "30a.mp4": 1, "30b.mp4": 1, "31.mp4": 1, "32.mp4": 1, "36a.mp4": 0,
    "36b.mp4": 1, "36c.mp4": 1, "42.mp4": 0, "47.mp4": 0, "50.mp4": 0,
    "51.mp4": 0, "52.mp4": 0, "53.mp4": 1, "54.mp4": 0, "55.mp4": 0,
    "56.mp4": 0, "57.mp4": 1, "59.mp4": 0, "60.mp4": 0, "62.mp4": 1,
    "63.mp4": 0, "65.mp4": 0, "66.mp4": 0, "67a.mp4": 1, "67b.mp4": 1,
    "68.mp4": 1, "69.mp4": 1, "70.mp4": 1, "71.mp4": 1, "72.mp4": 1,
    "75.mp4": 1, "76.mp4": 1, "78.mp4": 1, "80.mp4": 1, "82a.mp4": 1,
    "82b.mp4": 1, "83.mp4": 1, "84.mp4": 1, "85.mp4": 1, "87.mp4": 1,
    "88.mp4": 1, "89.mp4": 1, "90.mp4": 1, "92.mp4": 1, "93.mp4": 1
}

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

# Prepare dataset
features = []
labels = []

for video_file, label in target_labels.items():
    video_file_path = os.path.join(video_path, video_file)
    if os.path.exists(video_file_path):
        features.append(extract_features(video_file_path))
        labels.append(label)

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Preprocess the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(model, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
