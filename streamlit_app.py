import streamlit as st
import os
import requests

# Set Streamlit page configuration
st.set_page_config(page_title="Medical Diagnosis System", page_icon="🩺", layout="wide")

# Define GitHub Release URLs for models (Replace these with actual release links)
MODEL_URLS = {
    "brain_tumor": "https://github.com/Ashutosh8709/Health-Disease-Diagnosis-system/releases/download/v1.01/brain_tumor.h5",
    "retinopathy": "https://github.com/Ashutosh8709/Health-Disease-Diagnosis-system/releases/download/v1.01/retinopathy.h5",
    "osteoarthritis": "https://github.com/Ashutosh8709/Health-Disease-Diagnosis-system/releases/download/v1.01/osteoarthritis.h5",
    "chest_xray": "https://github.com/Ashutosh8709/Health-Disease-Diagnosis-system/releases/download/v1.01/chest_xray.h5"
}

# Local folder to store downloaded models
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

# Define models and their associated images
models = {
    "brain_tumor": {"name": "🧠 Brain Tumor", "image": "static/images/img_3.jpg"},
    "retinopathy": {"name": "👁️ Diabetic Retinopathy", "image": "static/images/img_2.jpg"},
    "osteoarthritis": {"name": "🦴 Osteoarthritis", "image": "static/images/img_1.jpg"},
    "chest_xray": {"name": "🩺 Chest X-ray", "image": "static/images/img_4.jpg"}
}

# Page Heading
st.markdown("<h1 style='text-align: center; color: yellow;'>Medical Diagnosis System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Select a Diagnostic Model</h3>", unsafe_allow_html=True)

# Display models in a grid layout
cols = st.columns(2)
index = 0
for model_key, model_info in models.items():
    with cols[index % 2]:
        st.image(model_info["image"], caption=model_info["name"], use_container_width=True)
        
        if st.button(f"🔍 Upload for {model_info['name']}"):
            st.session_state["selected_model"] = model_key  # Store selected model
            download_model(model_key)  # Ensure the model is downloaded before proceeding
            st.switch_page("pages/streamlit_upload.py")  # ✅ Correct navigation to `upload.py`
    
    index += 1
