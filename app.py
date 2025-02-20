import streamlit as st
import requests
import numpy as np
from PIL import Image
import io

# Streamlit UI
st.set_page_config(page_title="Medical Diagnosis System", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
            font-family: 'Josefin Sans', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #FFA500;
        }
        .subtext {
            text-align: center;
            font-size: 1.5rem;
            color: #f1f1f1;
        }
        .uploaded-img {
            max-width: 80%;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(255, 255, 255, 0.2);
        }
    </style>
""", unsafe_allow_html=True)

# Animated heading
st.markdown('<div class="title">🩺 Medical Diagnosis System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Select a diagnostic model and upload an image</div>', unsafe_allow_html=True)

# Define available models
model_dict = {
    "Brain Tumor": "brain_tumor",
    "Diabetic Retinopathy": "retinopathy",
    "Osteoarthritis": "osteoarthritis",
    "Chest X-ray (COVID, Pneumonia, TB)": "chest_xray"
}

# Model selection
selected_model = st.selectbox("Select a diagnostic model:", list(model_dict.keys()))

# Upload image
uploaded_file = st.file_uploader("Upload an image for diagnosis", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True, output_format="JPEG")

    # Convert image to bytes for sending
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    # Submit button
    if st.button("Analyze Image 🚀"):
        st.info("🔍 Analyzing image... Please wait.")

        # API call to backend
        response = requests.post(
            "http://127.0.0.1:5000/predict",  # Change to deployed backend URL
            files={"image": img_bytes},
            data={"model_type": model_dict[selected_model]}
        )

        if response.status_code == 200:
            data = response.json()
            prediction = data.get("prediction", "Unknown")
            confidence = data.get("confidence", 0)

            # Display results
            st.success(f"🎯 Diagnosis: **{prediction}**")
            st.write(f"📈 Confidence: **{(confidence * 100):.2f}%**")

            # Show confidence scores per class
            st.subheader("Detailed Predictions:")
            for i, prob in enumerate(data.get("all_predictions", [])):
                st.progress(prob)  # Show probability bars
                st.text(f"{i}: {prob*100:.1f}%")

        else:
            st.error("❌ Error analyzing image. Try again!")

