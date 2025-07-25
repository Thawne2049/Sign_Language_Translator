import cv2
import numpy as np
import mediapipe as mp
import os
import json
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

#  dataset path
DATASET_PATH = "C:/Users/sayye/OneDrive/Desktop/Retrain_ASL"
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

#  labels (A-Z, 0-9, SPACE)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)] + ["space"]


def extract_landmarks(image, draw=True):
    """Extracts hand landmarks and optionally draws them on the image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            if draw:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            all_landmarks.extend(landmarks)

        while len(all_landmarks) < 42:  # Ensure 126 features (pad if only one hand detected)
            all_landmarks.append([0, 0, 0])

        return np.array(all_landmarks).flatten()

    return None


# Start video capture
cap = cv2.VideoCapture(0)

print("\nWelcome to Sign Language Data Collection!")
print("Press 'Q' anytime to quit.")
time.sleep(2)

while True:
    print("\nAvailable labels:", labels)
    label = input("\nEnter the letter/number/space you want to collect data for (or 'exit' to stop): ").strip()

    if label.lower() == 'exit':
        break
    elif label.upper() not in labels and label.lower() != "space":
        print("Invalid label! Please enter a valid letter (A-Z), number (0-9), or 'space'.")
        continue

    # Convert to uppercase for consistency
    label = label.upper() if label.upper() in labels else "space"

    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    print(f"\nPress 'C' to start collecting 500 samples for '{label}'.")

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
