import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import os

# Page configuration
st.set_page_config(
    page_title="Construction Site Object Detection",
    page_icon="🏗️",
    layout="wide"
)

# Title and description
st.title("🏗️ Construction Site Object Detection")
st.markdown("""
Upload an image to detect:
- 🚧 Construction machinery and vehicles
- 👷 People and workers
- ⛑️ Safety equipment (hardhats, masks, safety vests)
- ⚠️ Safety violations (NO-Hardhat, NO-Mask, NO-Safety Vest)
""")

# Sidebar for settings
st.sidebar.header("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
imgsz = st.sidebar.selectbox("Image Size", [640, 1280, 1920], index=2)

# Model path
MODEL_PATH = "runs/detect/y11x_10ep/weights/best.pt"

# Load model (cached)
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Drag and drop or click to browse"
)

if uploaded_file is not None:
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Original Image")
        # Read image
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        # Display image info
        st.caption(f"Image size: {image.size[0]}x{image.size[1]} pixels")
    
    # Process button
    if st.button("🔍 Process Image", type="primary", use_container_width=True):
        with st.spinner("Processing image..."):
            # Convert PIL image to OpenCV format
            img_array = np.array(image)
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Run inference
            results = model.predict(
                source=img_array,
                imgsz=imgsz,
                conf=conf_threshold,
                augment=True,
                verbose=False
            )
            
            # Get detection results
            result = results[0]
            
            # Draw bounding boxes on image
            annotated_img = result.plot()
            
            # Convert back to RGB for display
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("✅ Detected Objects")
                st.image(annotated_img_rgb, use_container_width=True)
                
                # Display detection statistics
                num_detections = len(result.boxes)
                st.caption(f"Total detections: {num_detections}")
                
                # Count detections by class
                if num_detections > 0:
                    class_counts = {}
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    st.subheader("📊 Detection Summary")
                    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"- **{class_name}**: {count}")
                    
                    # Detailed detections
                    with st.expander("📋 View Detailed Detections"):
                        for i, box in enumerate(result.boxes, 1):
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            conf = float(box.conf[0])
                            st.write(f"{i}. **{class_name}** (Confidence: {conf:.2%})")
                
                # Download button
                img_bytes = io.BytesIO()
                Image.fromarray(annotated_img_rgb).save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                st.download_button(
                    label="💾 Download Result",
                    data=img_bytes,
                    file_name=f"detected_{uploaded_file.name}",
                    mime="image/png",
                    use_container_width=True
                )

else:
    # Show example message
    st.info("👆 Upload an image to get started!")
    
    # Show some tips
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        - **Higher confidence threshold** (0.4-0.6) = fewer but more certain detections
        - **Lower confidence threshold** (0.15-0.3) = more detections but may include false positives
        - **Larger image size** (1920) = better detection of small objects but slower processing
        - **Smaller image size** (640) = faster processing but may miss small objects
        - Works best with clear, well-lit construction site images
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This app uses **YOLO11x** fine-tuned on 2,603 construction site images.

**Model Performance:**
- mAP50-95: 38.2%
- mAP50: 73.8%
- Training time: 33 minutes (10 epochs)
""")
