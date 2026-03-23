import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ----------------------------
# 1️⃣ Load Model
# ----------------------------
model = load_model("model/landslide_model.h5")

# ----------------------------
# 2️⃣ Streamlit UI
# ----------------------------
st.set_page_config(page_title="🌍 Landslide Detection", layout="centered")
st.title("🌍 Landslide Detection App")
st.write("Upload a satellite image to check if it shows a landslide area.")

uploaded_file = st.file_uploader("Upload Image (jpg/png)", type=["jpg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(image)/255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    # Make prediction
    prediction = model.predict(img_array)[0][0]

    # Display result
    if prediction > 0.5:
        st.error("⚠️ Landslide Detected")
    else:
        st.success("✅ No Landslide Detected")