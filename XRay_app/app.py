# app.py
import streamlit as st
from PIL import Image
import torch
from utils.model_utils import load_model, predict
from utils.preprocessing import preprocess_image

st.set_page_config(page_title="X-ray Diagnosis Demo", layout="centered")
st.title("🩻 X-ray Multi-Label Diagnosis App (CheXNet)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/projectnb/dl4ds/projects/dca_project/scripts/models/59o12z7z-distinctive-snowflake-21/distinctive-snowflake-21.pth"
model = load_model(model_path, device)

uploaded_file = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    img_tensor = preprocess_image(image)

    probs = predict(model, img_tensor, device)

    st.subheader("Predictions")
    for disease, prob in probs.items():
        st.write(f"**{disease}**: {prob:.4f}")
