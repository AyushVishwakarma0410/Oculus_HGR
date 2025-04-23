import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from joblib import dump
import os

# === Load data ===
data_path = "C:\\xtra\\Last_Chance\\Oculus_HGR\\data\\merged_hand_gesture_data_common_labels.csv"
df = pd.read_csv(data_path)

X = df.drop(columns="label")
y = df["label"]

# === Encode labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# === Train model ===
model = XGBClassifier(eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring="accuracy")
print(f"[+] CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

y_pred = model.predict(X_test)
print("\n[+] Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === Save artifacts ===
os.makedirs("models", exist_ok=True)
dump(model, "models/gesture_model.joblib")
dump(scaler, "models/scaler.joblib")
dump(le, "models/label_encoder.joblib")

print("[+] Model, scaler, and label encoder saved to /models/")
