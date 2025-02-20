import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Ensure a model is selected
if "selected_model" not in st.session_state:
    st.error("No model selected! Please go back and select a model first.")
    st.stop()

# Load model details
model_key = st.session_state["selected_model"]
model_details = {
    "brain_tumor": {"name": "Brain Tumor", "model_path": "models/brain_tumor.h5", "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary"]},
    "retinopathy": {"name": "Diabetic Retinopathy", "model_path": "models/retinopathy.h5", "classes": ["No DR", "DR"]},
    "osteoarthritis": {"name": "Osteoarthritis", "model_path": "models/osteoarthritis.h5", "classes": ["Normal", "Osteopenia", "Osteoporosis"]},
    "chest_xray": {"name": "Chest X-ray", "model_path": "models/chest_xray.h5", "classes": ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]}
}

# Load the selected model
model_info = model_details[model_key]
model = load_model(model_info["model_path"])

# UI
st.markdown(f"<h1 style='text-align: center; color: yellow;'>Upload Image for {model_info['name']} Diagnosis</h1>", unsafe_allow_html=True)

# 🔙 Back to Home Button
if st.button("🔙 Back to Home"):
    st.switch_page("app.py")  # ✅ Redirects back to Home (app.py)

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
