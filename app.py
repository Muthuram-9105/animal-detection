import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO(r"runs\detect\detected2\weights\best.pt")  # Replace with your actual path

# Streamlit UI setup
st.set_page_config(page_title="Animal Detection - Farm Protection", layout="centered")
st.title("🦌 Animal Detection in Farm")
st.subheader("Upload an image to check for animals")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Placeholder for result
result_placeholder = st.empty()

def detect_elephant(image):
    # Set confidence threshold to 40%
    results = model(image, conf=0.6)
    names = model.names
    boxes = results[0].boxes

    found_elephant = False
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = names[class_id]
        if class_name.lower() == "elephant":
            found_elephant = True
            break

    # Annotated image with bounding boxes
    annotated_img = results[0].plot()
    return annotated_img, found_elephant

# Run detection
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)

    annotated_image, elephant_detected = detect_elephant(image_np)

    if elephant_detected:
        result_placeholder.success("✅ Animal detected in the farm (Elephant)")
    else:
        result_placeholder.error("❌ Wrong Detection")

    st.image(annotated_image, caption="Detection Result", use_column_width=True)
