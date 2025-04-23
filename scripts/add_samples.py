import cv2
import csv
import os
import numpy as np
import mediapipe as mp

# === Settings ===
gesture_label = input("Enter the gesture label (e.g., thumbs_up): ").strip()
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{gesture_label}.csv")

num_samples = 200  # Adjust how many samples you want to collect

# === Setup MediaPipe Hands ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# === Setup Webcam ===
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam. Press 's' to start collecting samples. Press 'q' to quit.")

collecting = False
collected = 0

with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
    writer.writerow(header)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if collecting and collected < num_samples:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    landmarks.append(gesture_label)
                    writer.writerow(landmarks)
                    collected += 1
                    print(f"[INFO] Collected sample {collected}/{num_samples}")

                elif collected >= num_samples:
                    collecting = False
                    print("[INFO] Done collecting samples.")

        cv2.putText(frame, f"Gesture: {gesture_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if collecting:
            cv2.putText(frame, f"Collecting: {collected}/{num_samples}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Collect Samples", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            collecting = True
            collected = 0
            print("[INFO] Started collecting samples...")
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
