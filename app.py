import logging
import queue
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from tensorflow.keras.models import load_model

# --- Streamlit UI ---
st.set_page_config(page_title="ASL Real-Time Detection", layout="centered")
st.title("ASL Real-Time Detection")

logger = logging.getLogger(__name__)

@st.cache_resource
def load_trained_model():
    return load_model("best_asl_model.h5")

model = load_trained_model()
labels = [chr(i) for i in range(65, 91)] 

class Detection(NamedTuple):
    label: str
    score: float

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    frame = cv2.flip(image, 1)

    h, w, _ = frame.shape

    # Kotak di kiri atas tapi agak ke bawah supaya teks terlihat
    box_size = 200
    x1 = 20  # margin kiri
    y1 = 80  # geser ke bawah dari atas
    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[y1:y2, x1:x2]
    pred_letter = '?'
    score = 0.0

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

    # Gambar kotak dan label prediksi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f'{pred_letter} ({score:.2f})',
        (x1, y2 + 30),  # teks di bawah kotak
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.putText(
        frame,
        'Arahkan tangan ke kotak',
        (x1, y2 + 70),  # teks instruksi di bawah teks prediksi
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    while not result_queue.empty():
        result_queue.get()
    result_queue.put([Detection(label=pred_letter, score=score)])

    return av.VideoFrame.from_ndarray(frame, format="bgr24")



# --- Streamer Setup ---
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

if st.checkbox("Tampilkan hasil prediksi", value=True):
    if webrtc_ctx.state.playing and not result_queue.empty():
        result = result_queue.get()
        st.table(result)
