import cv2
import numpy as np
import mediapipe as mp
import os
import json
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Define dataset path
DATASET_PATH = "C:/Users/sayye/OneDrive/Desktop/Retrain_ISL"
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# Define labels (A-Z, 0-9)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)]


def extract_landmarks(image, draw=True):
    """Extracts hand landmarks with exception handling and extra padding."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    landmarks_list = []  # Stores landmarks for both hands

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if draw:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_landmarks_list = []
            for lm in hand_landmarks.landmark:
                hand_landmarks_list.append([lm.x, lm.y, lm.z])

            landmarks_list.append(hand_landmarks_list)

    # Ensure at least 126 features (63 per hand)
    while len(landmarks_list) < 2:  # If only one hand detected, pad the second hand
        landmarks_list.append([[0, 0, 0]] * 21)

    if len(landmarks_list) == 0:  # If no hands detected, return None
        return None

    # Flatten to 126 features
    full_features = np.array(landmarks_list).flatten()

    # *Add extra padding to reach 150 features for safety*
    if len(full_features) < 150:
        full_features = np.pad(full_features, (0, 150 - len(full_features)), mode='constant')

    return full_features[:150]  # Ensure 150 features per sample


# Start video capture
cap = cv2.VideoCapture(0)

print("\nWelcome to Sign Language Data Collection!")
print("Press 'Q' anytime to quit.")
time.sleep(2)

while True:
    print("\nAvailable labels:", labels)
    label = input("\nEnter the letter/number you want to collect data for (or 'exit' to stop): ").strip().upper()

    if label.lower() == 'exit':
        break
    elif label not in labels:
        print("Invalid label! Please enter a valid letter (A-Z) or number (0-9).")
        continue

    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    print(f"\nPress 'C' to start collecting 200 samples for '{label}'.")

    samples_collected = 0
    collecting = False  # Flag to track whether to collect samples

    while samples_collected < 500:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror effect
        landmarks = extract_landmarks(frame, draw=True)

        # Display instructions on screen
        if not collecting:
            cv2.putText(frame, "Press 'C' to start capturing", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        if collecting and landmarks is not None:
            file_path = os.path.join(label_path, f"{samples_collected}.json")
            with open(file_path, "w") as f:
                json.dump(landmarks.tolist(), f)

            samples_collected += 1
            print(f"Collected: {samples_collected}/500 for {label}")

        # Display progress
        cv2.putText(frame, f"Collecting {label}: {samples_collected}/500", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'Q' to quit", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif key == ord('c') and not collecting:
            collecting = True  # Start capturing when 'C' is pressed

cap.release()
cv2.destroyAllWindows()
print("\nData collection complete!")