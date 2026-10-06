import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AI Image Classification",
    page_icon="🖼️",
    layout="centered"
)

# Title
st.title("🖼️ AI-Based Image Classification System")
st.write("Upload an image and the AI model will identify the objects in it.")

# Load YOLO classification model
@st.cache_resource
def load_model():
    return YOLO("yolo11n-cls.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# Confidence threshold
confidence = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display original image
    st.subheader("📷 Input Image")
    st.image(image, use_container_width=True)

    # Convert image to NumPy array
    image_array = np.array(image)

    # Perform classification
    results = model(image_array)

    # Get classification result
    result = results[0]

    # Get probabilities
    probabilities = result.probs

    if probabilities is not None:

        # Top prediction
        top_class = int(probabilities.top1)
        top_confidence = float(probabilities.top1conf)

        class_name = model.names[top_class]

        st.subheader("🤖 Classification Result")

        if top_confidence >= confidence:
            st.success(
                f"Prediction: **{class_name}**"
            )

            st.info(
                f"Confidence: **{top_confidence:.2%}**"
            )

        else:
            st.warning(
                "The model is not confident enough."
            )

        # Show top 5 predictions
        st.subheader("🔍 Top Predictions")

        top5_indices = probabilities.top5
        top5_confidences = probabilities.top5conf

        for i in range(len(top5_indices)):

            class_id = int(top5_indices[i])
            class_name = model.names[class_id]
            score = float(top5_confidences[i])

            st.write(
                f"**{i + 1}. {class_name}** — {score:.2%}"
            )

    else:
        st.error("Unable to classify the image.")
