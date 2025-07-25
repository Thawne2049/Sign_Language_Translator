import cv2
import os
import mediapipe as mp

# Ask the user for the folder name to save images
gesture_name = input("Enter the name of the gesture (e.g., 'A_sign'): ").strip()
save_dir = os.path.join("captured_hand_images", gesture_name)

# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Start video capture
cap = cv2.VideoCapture(0)
capturing = False  # Flag to check if capturing is active
image_count = 0  # Counter for images

print("\nPress 'C' to start capturing images, and 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for natural view
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Mediapipe
    results = hands.process(rgb_frame)  # Detect hands

    # Draw hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if capturing:
            # Save the image with the given gesture name
            filename = os.path.join(save_dir, f"{gesture_name}_{image_count}.jpg")
            cv2.imwrite(filename, frame)
            image_count += 1
            cv2.putText(frame, "Capturing...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the video feed
    cv2.imshow("Hand Capture", frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        capturing = not capturing  # Toggle capturing mode
        print(f"Capturing {'started' if capturing else 'stopped'}...")
    elif key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Saved {image_count} images in '{save_dir}'.")
