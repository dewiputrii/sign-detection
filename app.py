import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# Load model
@st.cache_resource
def load_trained_model():
    return load_model('best_asl_model.h5')

model = load_trained_model()
labels = [chr(i) for i in range(65, 91)]

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

st.title("ASL Real-Time Detection (Hand Tracking)")
FRAME_WINDOW = st.image([])
start = st.button("Start")
stop = st.button("Stop")

if start:
    cap = cv2.VideoCapture(0)
    stop_flag = False

    while cap.isOpened() and not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get bounding box from landmarks
                h, w, _ = frame.shape
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]
                x_min = int(min(x_list) * w) - 20
                y_min = int(min(y_list) * h) - 20
                x_max = int(max(x_list) * w) + 20
                y_max = int(max(y_list) * h) + 20

                # Ensure boundaries stay within frame
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)

                roi = frame[y_min:y_max, x_min:x_max]

                # Prediction if ROI is valid
                if roi.size != 0:
                    try:
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (28, 28))
                        normalized = resized / 255.0
                        reshaped = normalized.reshape(1, 28, 28, 1)
                        pred = model.predict(reshaped)
                        pred_letter = labels[np.argmax(pred)]
                    except:
                        pred_letter = '?'

                    cv2.putText(frame, f'Prediction: {pred_letter}', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            cv2.putText(frame, 'No Hand Detected', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

        if stop:
            stop_flag = True
            break

        time.sleep(0.03)

    cap.release()
    st.success("Camera Stopped")
