import streamlit as st 
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import pandas as pd
import os
import time
from difflib import SequenceMatcher
import base64

# --- PAGE CONFIG ---
st.set_page_config(page_title="Car License Plate Detection", layout="centered")

# --- BASE64 BACKGROUND IMAGE ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("assets/car3.jpg")

# --- CUSTOM CSS STYLING ---
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }}

    .main-container {{
        background-color: rgba(0, 0, 0, 0.5);
        padding: 50px 40px;
        text-align: center;
        max-width: 600px;
        margin: 100px auto 30px auto;
        border-radius: 12px;
    }}

    .main-title {{
        font-size: 32px;
        font-weight: bold;
        color: #87CEEB;  /* Sky Blue */
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8); /* Glow effect */
    }}

    .upload-btn {{
        display: inline-block;
        background-color: #008CFF;
        color: white;
        padding: 10px 30px;
        font-size: 16px;
        border-radius: 6px;
        margin-top: 15px;
        text-decoration: none;
        cursor: pointer;
    }}

    .contact {{
        text-align: center;
        font-size: 16px;
        margin-top: 60px;
        font-weight: bold;
    }}

    .contact a {{
        color: #42A5F5;
        text-decoration: none;
    }}

    /* Button styling */
    div.stButton > button:first-child {{
        background-color: #1E88E5;
        color: white;
        border-radius: 6px;
        padding: 8px 20px;
    }}

    /* File uploader */
    .stFileUploader {{
        background-color: rgba(255, 255, 255, 0.1);
        border: 2px solid #64B5F6;
        border-radius: 10px;
        padding: 10px;
    }}

    /* Tables */
    .stDataFrame tbody td {{
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
    }}
    </style>
""", unsafe_allow_html=True)

# --- MAIN INTERFACE CONTAINER ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="main-title">Car License Plate Detection</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    st.markdown('<div class="upload-btn">Uploading...</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close main container


# --- Load Models ---
model = YOLO('best.pt')
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# --- Setup temp directory ---
os.makedirs("temp", exist_ok=True)

# --- Helper Functions ---
def is_similar(text, seen_texts, threshold=0.85):
    for seen in seen_texts:
        if SequenceMatcher(None, text, seen).ratio() >= threshold:
            return True
    return False

def extract_text_from_region(region, ocr_engine, seen_texts):
    result = ocr_engine.ocr(region, cls=True)
    extracted = []
    if result:
        for line in result:
            if not line:
                continue
            for item in line:
                if item and len(item) == 2:
                    (bbox, (text, conf)) = item
                    if not is_similar(text, seen_texts) and text.strip():
                        seen_texts.add(text)
                        extracted.append({
                            'Detected Text': text,
                            'Confidence': f"{conf * 100:.2f}%"
                        })
    return extracted

def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(image_rgb, device='cpu')

    ocr_data = []
    seen_texts = set()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            roi = image_rgb[y1:y2, x1:x2]
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            ocr_data.extend(extract_text_from_region(roi_bgr, ocr, seen_texts))

    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return output_path, ocr_data

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return None, []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ocr_data_all = []
    seen_texts = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb_frame, device='cpu')

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    ocr_data_all.extend(extract_text_from_region(roi, ocr, seen_texts))

        out.write(frame)

    cap.release()
    out.release()
    time.sleep(1)
    return output_path, ocr_data_all

def process_media(input_path, output_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        return process_image(input_path, output_path)
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        return process_video(input_path, output_path)
    else:
        st.error("Unsupported file type.")
        return None, []

# --- Main Logic ---
if uploaded_file is not None:
    input_path = os.path.join("temp", uploaded_file.name)
    output_path = os.path.join("temp", f"output_{uploaded_file.name}")

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ File saved: {input_path}")
    st.write("🚀 Processing...")

    with st.spinner("Running object detection and OCR..."):
        result_path, ocr_results = process_media(input_path, output_path)

    if result_path:
        if result_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            with open(result_path, 'rb') as f:
                video_bytes = f.read()
            st.video(video_bytes)
        else:
            st.image(result_path, use_container_width=True)

        if ocr_results:
            st.subheader("🔍 Unique OCR Detected Texts")
            df = pd.DataFrame(ocr_results)
            st.dataframe(df)
        else:
            st.info("No text detected.")


# --- CONTACT INFO ---
st.markdown('<div class="contact">Contact: <a href="mailto:shobanbabujatoth@gmail.com">shobanbabujatoth@gmail.com</a></div>', unsafe_allow_html=True)




