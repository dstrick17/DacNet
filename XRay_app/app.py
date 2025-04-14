import streamlit as st
from PIL import Image
import torch
from utils.model_utils import load_model, predict
from utils.preprocessing import preprocess_image

st.set_page_config(page_title="X-ray Diagnosis Demo", layout="centered")
st.title("🩻 X-ray Multi-Model Diagnostic App (Demo)")

uploaded_file = st.file_uploader("Upload an X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    img_tensor = preprocess_image(image)

    # Dummy model names (can be changed later)
    model_names = ["Model A", "Model B", "Model C"]

    for model_name in model_names:
        model = load_model()  # uses dummy model
        probs = predict(model, img_tensor)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

        st.markdown(f"### 🤖 {model_name}")
        st.write(f"**Prediction:** Class {pred_class}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
