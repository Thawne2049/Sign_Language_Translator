import cv2
import numpy as np
import mediapipe as mp
import joblib
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import pyttsx3
import threading
import webbrowser
import os

# Default model paths
models = {
    "ASL": ("sign_language_model.h5", "label_encoder.pkl"),
    "FSL": ("sign_language_model.h5", "label_encoder.pkl"),
    "ISL": ("sign_language_model_isl.h5", "label_encoder_isl.pkl"),
    "BSL": ("sign_language_model_isl.h5", "label_encoder_isl.pkl"),
}

current_language = "ASL"
asl_model = tf.keras.models.load_model(models[current_language][0])
asl_label_encoder = joblib.load(models[current_language][1])

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Tkinter GUI Setup
root = tk.Tk()
root.title("Sign Language Translator")
root.geometry("1980x1080")
root.configure(bg="Light Gray")

# Global variables
captured_text = ""
stop_camera = False
capture_enabled = False
tts_mode_enabled = False  # Text-to-speech mode toggle
animation_window = None

def extract_landmarks(image):
    """Extracts hand landmarks from a video frame."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    all_landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                all_landmarks.append([lm.x, lm.y, lm.z])

    flattened_landmarks = np.array(all_landmarks).flatten()
    if len(flattened_landmarks) < 126:
        flattened_landmarks = np.pad(flattened_landmarks, (0, 126 - len(flattened_landmarks)))

    return flattened_landmarks if len(all_landmarks) > 0 else None

def predict_sign(frame):
    """Predicts the sign language gesture from the frame."""
    landmarks = extract_landmarks(frame)
    if landmarks is not None:
        landmarks = landmarks.reshape(1, -1)
        prediction = asl_model.predict(landmarks)
        predicted_label = asl_label_encoder.inverse_transform([np.argmax(prediction)])[0]
        return predicted_label
    return None

def update_camera():
    """Updates the live webcam feed."""
    global stop_camera, capture_enabled, captured_text

    ret, frame = cap.read()
    if ret and not stop_camera:
        frame = cv2.flip(frame, 1)
        predicted_label = predict_sign(frame)

        display_text = predicted_label if predicted_label else "No Hands Detected"
        cv2.putText(frame, f"Sign: {display_text}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)

        if capture_enabled and predicted_label:
            if predicted_label == "space":
                captured_text += " "
            else:
                captured_text += predicted_label

            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, captured_text)

            if tts_mode_enabled:
                engine.say(predicted_label)  # Speak the captured letter
                engine.runAndWait()
                capture_enabled = False  # Capture only once when in TTS Mode

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (500, 400))
        img_tk = tk.PhotoImage(data=cv2.imencode('.png', img)[1].tobytes())

        cam_label.img_tk = img_tk
        cam_label.config(image=img_tk)

    root.after(10, update_camera)

def toggle_tts_mode():
    """Toggles text-to-speech capture mode."""
    global tts_mode_enabled
    tts_mode_enabled = not tts_mode_enabled
    tts_toggle_btn.config(text=f"TTS Mode: {'ON' if tts_mode_enabled else 'OFF'}")

def capture_text(event=None):
    """Captures text when 'C' is pressed."""
    global capture_enabled
    capture_enabled = True

def speak_text():
    """Speaks the captured text."""
    global captured_text
    if captured_text.strip():
        threading.Thread(target=lambda: engine.say(captured_text) or engine.runAndWait()).start()

def clear_text():
    """Clears the text output area."""
    global captured_text
    captured_text = ""
    text_output.delete("1.0", tk.END)

def show_animation():
    """Displays animation for the manually entered letter/word."""
    global animation_window

    text = manual_entry.get().upper()
    video_path = f"assets/{text}.mp4"

    if os.path.exists(video_path):
        if animation_window:
            animation_window.destroy()

        animation_window = tk.Toplevel(root)
        animation_window.title(f"Sign Animation - {text}")
        animation_window.geometry("500x500")

        video_label = tk.Label(animation_window)
        video_label.pack()

        cap_animation = cv2.VideoCapture(video_path)

        def play_video():
            while cap_animation.isOpened():
                ret, frame = cap_animation.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (400, 400))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())

                video_label.img_tk = frame
                video_label.config(image=frame)
                animation_window.update()

            cap_animation.release()

        threading.Thread(target=play_video).start()

def change_language(event):
    """Handles language change from dropdown."""
    global current_language, asl_model, asl_label_encoder
    selected = language_selector.get()
    current_language = selected
    asl_model = tf.keras.models.load_model(models[selected][0])
    asl_label_encoder = joblib.load(models[selected][1])
    messagebox.showinfo("Language Changed", f"Language switched to {selected}")

# GUI Layout
title_label = tk.Label(root, text="Sign Language Translator", font=("Arial", 20, "bold"), bg="white", fg="black")
title_label.pack(pady=10)

# Language Selector
language_selector = ttk.Combobox(root, values=list(models.keys()), state="readonly", font=("Arial", 12))
language_selector.set(current_language)
language_selector.pack(pady=5)
language_selector.bind("<<ComboboxSelected>>", change_language)

cam_label = tk.Label(root, bg="black")
cam_label.pack()

text_output = scrolledtext.ScrolledText(root, width=40, height=5, font=("Arial", 14))
text_output.pack(pady=10)

manual_entry = tk.Entry(root, font=("Arial", 14))
manual_entry.pack(pady=5)

show_animation_btn = tk.Button(root, text="Show Animation", command=show_animation)
show_animation_btn.pack(pady=5)

tts_speak_frame = tk.Frame(root, bg="Light Gray")
tts_speak_frame.pack(pady=5)

tts_toggle_btn = tk.Button(tts_speak_frame, text="TTS Mode: OFF", command=toggle_tts_mode)
tts_toggle_btn.pack(side=tk.LEFT, padx=10)

speak_btn = tk.Button(tts_speak_frame, text="Speak", command=speak_text)
speak_btn.pack(side=tk.LEFT, padx=10)

clear_btn = tk.Button(tts_speak_frame, text="Clear", command=clear_text)
clear_btn.pack(side=tk.LEFT, padx=10)
root.bind("<c>", capture_text)

# --- Horizontal layout for Webcam and Jitsi ---
main_frame = tk.Frame(root)
main_frame.pack(pady=10)

# --- Left Side: Webcam Feed ---
cam_label = tk.Label(main_frame, bg="black")
cam_label.pack(side=tk.LEFT, padx=10)

# --- Right Side: Jitsi Video Frame ---
video_frame = tk.LabelFrame(main_frame, text="Video Calling (Jitsi)", font=("Arial", 12, "bold"), bg="lightgray", fg="black")
video_frame.pack(side=tk.LEFT, padx=10)

meeting_label = tk.Label(video_frame, text="Enter Room Name:", font=("Arial", 11), bg="lightgray")
meeting_label.pack(pady=5)

meeting_entry = tk.Entry(video_frame, font=("Arial", 12), width=30)
meeting_entry.pack(pady=5)

def create_meeting():
    room_name = meeting_entry.get().strip()
    if room_name:
        webbrowser.open(f"https://meet.jit.si/{room_name}")
    else:
        messagebox.showwarning("Missing Room", "Please enter a room name.")

def join_meeting():
    room_name = meeting_entry.get().strip()
    if room_name:
        webbrowser.open(f"https://meet.jit.si/{room_name}")
    else:
        messagebox.showwarning("Missing Room", "Please enter a room name to join.")

btn_frame = tk.Frame(video_frame, bg="lightgray")
btn_frame.pack(pady=10)

create_btn = tk.Button(btn_frame, text="Create Meeting", font=("Arial", 11), bg="#4CAF50", fg="white", command=create_meeting)
create_btn.pack(side=tk.LEFT, padx=10)

join_btn = tk.Button(btn_frame, text="Join Meeting", font=("Arial", 11), bg="#2196F3", fg="white", command=join_meeting)
join_btn.pack(side=tk.LEFT, padx=10)

update_camera()
root.mainloop()
