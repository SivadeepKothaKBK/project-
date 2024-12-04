import pandas as pd
import numpy as np
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

