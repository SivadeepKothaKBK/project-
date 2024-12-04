import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib

# Import the models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# Load the dataset
data_path = "C:\\Users\\sivad\\OneDrive\\Desktop\\project\\datasets\\cleaned_fraud_data.csv"
data = pd.read_csv(data_path)

# Feature selection
features = ["distance_from_home", "distance_from_last_transaction",
            "ratio_to_median_purchase_price", "repeat_retailer", 
            "used_chip", "used_pin_number", "online_order"]
X = data[features]
y = data['fraud']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                    test_size=0.2, random_state=42, stratify=y_resampled)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define all models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0),
    'Neural Network': MLPClassifier(max_iter=100, early_stopping=True),
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='hist'),
    'LightGBM': lgb.LGBMClassifier()
}

# Evaluate each model and store performance metrics
best_model = None
best_auc = 0
results = []

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_scaled, y_train)  # Train the model
    y_pred = model.predict(X_test_scaled)  # Predict on the test set
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append((model_name, accuracy, precision, recall, f1, auc))
    
    # Print performance for each model
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save the best model based on AUC
    if auc > best_auc:
        best_auc = auc
        best_model = model

# Save the best model and scaler
joblib.dump(best_model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Print the best model
print("\nBest Model:")
print(f"Model: {best_model.__class__.__name__}, AUC: {best_auc:.4f}")

# Save results as a DataFrame for review
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'])
results_df.to_csv('model_results.csv', index=False)  # Save model performance

# Print all models' results
print("\nAll Model Performance Metrics:")
print(results_df)

# Path to video files
video_path = r"C:\\Users\\sivad\\OneDrive\\Desktop\\project\\videos\\videos"

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
video_model = RandomForestClassifier(n_estimators=100, random_state=42)
video_model.fit(X_train, y_train)

# Evaluate the model
y_pred = video_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the video model and scaler
joblib.dump(video_model, 'fraud_detection_video_model.pkl')
joblib.dump(scaler, 'video_scaler.pkl')
