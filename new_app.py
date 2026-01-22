# -*- coding: utf-8 -*-
"""
Corn Leaf Disease Detection using Vision Transformers + SVM
"""

import os
import pickle
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "vit_model.pkl")

# ---------------------------
# Load trained SVM classifier
# ---------------------------
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file vit_model.pkl not found. Please upload it to your GitHub repo.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    classifier = pickle.load(f)

# ---------------------------
# Load ViT (feature extractor)
# ---------------------------
processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)
vit = ViTModel.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)
vit.eval()

# ---------------------------
# Class names (MUST match training)
# ---------------------------
CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="Corn Leaf Disease Detection",
    layout="centered"
)

# ---------------------------
# Custom Styling
# ---------------------------
st.markdown("""
<style>
.stApp {
    background: url('https://www.plants-wallpapers.com/plant/trees-corn-corn-orchard.jpg') no-repeat center center fixed;
    background-size: cover;
}
h1, p, label {
    color: white !important;
    text-align: center;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    border-radius: 8px;
}
.stButton>button:hover {
    background-color: #45a049;
}
.prediction-box {
    background-color: rgba(0, 0, 0, 0.65);
    color: #ffffff !important;
    padding: 16px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    margin-top: 15px;
}
.confidence-box {
    background-color: rgba(0, 0, 0, 0.55);
    color: #ffffff !important;
    padding: 12px;
    border-radius: 8px;
    font-size: 18px;
    text-align: center;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# UI
# ---------------------------
st.markdown("<h1>🌽 Corn Leaf Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload a corn leaf image to predict disease</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

predict_btn = st.button("Predict Disease")

# ---------------------------
# Prediction
# ---------------------------
if predict_btn:

    if uploaded_file is None:
        st.error("Please upload an image first.")
        st.stop()

    # ---------------------------
    # Read image using PIL (NO OpenCV)
    # ---------------------------
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    # ---------------------------
    # Extract ViT features
    # ---------------------------
    inputs = processor(images=pil_img, return_tensors="pt")

    with torch.no_grad():
        outputs = vit(**inputs)
        features = outputs.last_hidden_state[:, 0, :].numpy()
        # shape: (1, 768)

    # ---------------------------
    # Predict using SVM
    # ---------------------------
    pred_idx = classifier.predict(features)[0]
    confidence = np.max(classifier.predict_proba(features)) * 100
    result = CLASS_NAMES[pred_idx]

    # ---------------------------
    # Display result
    # ---------------------------
    st.markdown(
        f"<div class='prediction-box'>🌿 Prediction: {result}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='confidence-box'>Confidence: {confidence:.2f}%</div>",
        unsafe_allow_html=True
    )
