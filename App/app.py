import streamlit as st
import cv2
import joblib
import numpy as np
import mediapipe as mp

# Load model
model = joblib.load("Model/xgb_model.pkl")

st.title("Hand Gesture Recognition")

# Label mapping (make sure order matches your training!)
label_map = {
    0: "open_palm",
    1: "fist",
    2: "thumbs up"
}

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7) as hands:
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            return np.array(landmarks).reshape(1, -1)  # shape (1,42)
    return None

# Webcam input
run = st.checkbox('Run Webcam')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        continue

    # Extract features (42 hand landmarks)
    features = extract_hand_landmarks(frame)

    if features is not None:
        prediction = model.predict(features)[0]
        gesture = label_map.get(prediction, "Unknown")  # Convert number -> label
        cv2.putText(frame, f'Prediction: {gesture}', (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No hand detected", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
