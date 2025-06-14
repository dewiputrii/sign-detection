import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# --- Setup ---
st.set_page_config(page_title="ASL Detection", layout="centered")

# --- Load Model ---
@st.cache_resource
def load_trained_model():
    return load_model('best_asl_model.h5')

model = load_trained_model()
labels = [chr(i) for i in range(65, 91)]  # A-Z

# --- MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# --- Streamlit UI ---
st.title("ASL Real-Time Detection")
FRAME_WINDOW = st.image([])

# --- Button State ---
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

start = st.button("📷 Start Camera")
stop = st.button("🛑 Stop Camera")

if start:
    st.session_state.camera_on = True

if stop:
    st.session_state.camera_on = False
    st.success("Camera stopped.")

# --- Webcam Streaming ---
if st.session_state.camera_on:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("🚫 Gagal mengakses kamera. Pastikan kamera tersedia dan tidak digunakan aplikasi lain.")
        st.session_state.camera_on = False
    else:
        
        for _ in range(200):
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Frame tidak terbaca.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            pred_letter = '?'

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    x_list = [lm.x for lm in hand_landmarks.landmark]
                    y_list = [lm.y for lm in hand_landmarks.landmark]
                    x_min = int(min(x_list) * w) - 20
                    y_min = int(min(y_list) * h) - 20
                    x_max = int(max(x_list) * w) + 20
                    y_max = int(max(y_list) * h) + 20

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)

                    roi = frame[y_min:y_max, x_min:x_max]

                    if roi.size != 0:
                        try:
                            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            resized = cv2.resize(gray, (28, 28))
                            normalized = resized / 255.0
                            reshaped = normalized.reshape(1, 28, 28, 1)
                            pred = model.predict(reshaped, verbose=0)
                            pred_letter = labels[np.argmax(pred)]
                        except Exception as e:
                            pred_letter = '?'

                    cv2.putText(frame, f'Prediction: {pred_letter}', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                cv2.putText(frame, 'No Hand Detected', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            time.sleep(0.03)

        cap.release()
