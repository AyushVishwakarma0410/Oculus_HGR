import cv2
import numpy as np
import mediapipe as mp
from joblib import load
import pandas as pd

# === Load trained model and preprocessors ===
model = load("models/gesture_model.joblib")
scaler = load("models/scaler.joblib")
label_encoder = load("models/label_encoder.joblib")

# === Setup MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# === Start webcam ===
cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand landmarks (63 values)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                # Prepare input
                X_input = pd.DataFrame([landmarks],
                    columns=[f"{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']])
                X_scaled = scaler.transform(X_input)

                # Predict
                prediction = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled).max()
                gesture = label_encoder.inverse_transform([prediction])[0]

                # Display prediction
                cv2.putText(frame, f"{gesture} ({proba:.2f})", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
