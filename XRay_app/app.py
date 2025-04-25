# app.py
import streamlit as st
from PIL import Image
import torch
from model_utils import load_model, predict,  generate_gradcam
from preprocessing import preprocess_image
import numpy as np
import cv2




st.set_page_config(page_title="X-ray Diagnosis Demo", layout="centered")
st.title("🩻 X-ray Multi-Label Diagnosis App (CheXNet)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device)


uploaded_file = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    img_tensor = preprocess_image(image)

    probs = predict(model, img_tensor, device)

    # Get top class
    top_disease = max(probs, key=probs.get)
    target_idx = list(probs.keys()).index(top_disease)

# Grad-CAM
    cam = generate_gradcam(model, img_tensor, target_idx, device)

# Overlay on image
    image_resized = image.resize((224, 224))
    img_np = np.array(image_resized)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

# Show it
    st.subheader(f"Grad-CAM Visualization: {top_disease}")
    st.image(overlay, use_container_width=True)


    st.subheader("Predictions")
    for disease, prob in probs.items():
        st.write(f"**{disease}**: {prob:.4f}")
