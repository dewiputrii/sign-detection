import logging
import queue
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from tensorflow.keras.models import load_model

st.set_page_config(page_title="ASL Real-Time Detection", layout="centered")
st.title("ASL Real-Time Detection")

logger = logging.getLogger(__name__)

@st.cache_resource
def load_trained_model():
    return load_model("best_asl_model.h5")

model = load_trained_model()
labels = [chr(i) for i in range(65, 91)]  # A-Z

class Detection(NamedTuple):
    label: str
    score: float

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def detect_hand_roi(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 5000:  # filter noise
            x, y, w, h = cv2.boundingRect(c)
            return frame[y:y+h, x:x+w], (x, y, w, h)
    return None, None

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    frame = cv2.flip(image, 1)

    roi, box = detect_hand_roi(frame)
    pred_letter = '?'
    score = 0.0

    if roi is not None:
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

        
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f'{pred_letter} ({score:.2f})',
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    
    while not result_queue.empty():
        result_queue.get()

    result_queue.put([Detection(label=pred_letter, score=score)])
    return av.VideoFrame.from_ndarray(frame, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="asl-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15}
        },
        "audio": False
    },
    async_processing=True,
)

# --- Result Table ---
if st.checkbox("Tampilkan hasil prediksi", value=True):
    if webrtc_ctx.state.playing and not result_queue.empty():
        result = result_queue.get()
        st.table(result)
