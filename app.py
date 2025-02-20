import streamlit as st

# Set Streamlit page configuration
st.set_page_config(page_title="Medical Diagnosis System", page_icon="🩺", layout="wide")

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
            st.switch_page("pages/upload.py")  # ✅ Correct path to the `upload.py` file
    
    index += 1
