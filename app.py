import logging
import queue
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import mediapipe as mp
from tensorflow.keras.models import load_model

st.set_page_config(page_title="ASL Real-Time Detection", layout="centered")
st.title("ASL Real-Time Detection")

logger = logging.getLogger(__name__)

@st.cache_resource
def load_trained_model():
    model = load_model('best_asl_model.h5')
    return model

model = load_trained_model()
labels = [chr(i) for i in range(65, 91)]  # A-Z

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

class Detection(NamedTuple):
    label: str
    score: float

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    frame = cv2.flip(image, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    pred_letter = '?'
    score = 0.0

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
                    score = float(np.max(pred))
                    pred_letter = labels[np.argmax(pred)]
                except Exception as e:
                    logger.warning(f"Prediction error: {e}")
                    pred_letter = '?'

            cv2.putText(
                frame,
                f'Prediction: {pred_letter}',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    result_queue.put([Detection(label=pred_letter, score=score)])
    return av.VideoFrame.from_ndarray(frame, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="asl-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
    # client_settings=ClientSettings(
    #     media_stream_constraints={"video": True, "audio": False},
    #     rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    # )
)

if st.checkbox("Tampilkan hasil prediksi", value=True):
    if webrtc_ctx.state.playing:
        if not result_queue.empty():
            result = result_queue.get()
            st.table(result)
