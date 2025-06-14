import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io
import json

model = load_model('best_asl_model.h5')
labels = [chr(i) for i in range(65, 91)]

st.set_page_config(page_title="ASL API", layout="wide")

query = st.experimental_get_query_params()

if "upload" in query:
    uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)

        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (28, 28))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 28, 28, 1)
        pred = model.predict(reshaped)
        pred_letter = labels[np.argmax(pred)]

        st.json({ "prediction": pred_letter })
else:
    st.markdown("### ASL Real-Time API\nGunakan endpoint `/upload` untuk prediksi gambar dari webcam frontend JS.")
