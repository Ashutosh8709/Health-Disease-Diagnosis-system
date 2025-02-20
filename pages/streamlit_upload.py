import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Ensure a model is selected
if "selected_model" not in st.session_state:
    st.error("No model selected! Please go back and select a model first.")
    st.stop()

# Define GitHub Release URLs for models (Replace with actual release links)
MODEL_URLS = {
    "brain_tumor": "https://github.com/Ashutosh8709/Health-Disease-Diagnosis-system/releases/download/v1.01/brain_tumor.h5",
    "retinopathy": "https://github.com/Ashutosh8709/Health-Disease-Diagnosis-system/releases/download/v1.01/retinopathy.h5",
    "osteoarthritis": "https://github.com/Ashutosh8709/Health-Disease-Diagnosis-system/releases/download/v1.01/osteoarthritis.h5",
    "chest_xray": "https://github.com/Ashutosh8709/Health-Disease-Diagnosis-system/releases/download/v1.01/chest_xray.h5"
}

# Local directory to store downloaded models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_model(model_name):
    """Download the model from GitHub Releases if not available locally."""
    model_path = os.path.join(MODEL_DIR, f"{model_name}.h5")
    
    if not os.path.exists(model_path):
        st.info(f"Downloading {model_name} model...")
        response = requests.get(MODEL_URLS[model_name], stream=True)
        
        if response.status_code == 200:
            with open(model_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            st.success(f"{model_name} model downloaded successfully!")
        else:
            st.error("Failed to download the model. Please check the URL.")
    
    return model_path  # Return the local path of the downloaded model

# Load model details
model_key = st.session_state["selected_model"]
model_details = {
    "brain_tumor": {"name": "Brain Tumor", "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary"]},
    "retinopathy": {"name": "Diabetic Retinopathy", "classes": ["No DR", "DR"]},
    "osteoarthritis": {"name": "Osteoarthritis", "classes": ["Normal", "Osteopenia", "Osteoporosis"]},
    "chest_xray": {"name": "Chest X-ray", "classes": ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]}
}

# Download model if not available
model_path = download_model(model_key)

# Load the selected model
model_info = model_details[model_key]
model = load_model(model_path)

# UI
st.markdown(f"<h1 style='text-align: center; color: yellow;'>Upload Image for {model_info['name']} Diagnosis</h1>", unsafe_allow_html=True)

# 🔙 Back to Home Button
if st.button("🔙 Back to Home"):
    st.switch_page("streamlit_app.py")  # ✅ Redirects back to Home (app.py)

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = model_info["classes"][np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display results
    st.success(f"🎯 **Diagnosis:** {predicted_class}")
    st.info(f"📈 **Confidence:** {confidence:.2f}%")

    # Show detailed predictions
    st.subheader("Detailed Predictions")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{model_info['classes'][i]}: {prob * 100:.2f}%")
