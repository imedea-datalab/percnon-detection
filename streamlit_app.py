import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64
import json
import zipfile
import os
import sahi
import hashlib
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

if "batch_key" not in st.session_state:
    st.session_state.batch_key = None

if "image_summary_df" not in st.session_state:
    st.session_state.image_summary_df = None

if "crab_summary_df" not in st.session_state:
    st.session_state.crab_summary_df = None

if "batch_predictions_zip" not in st.session_state:
    st.session_state.batch_predictions_zip = None

# Page configuration
st.set_page_config(
    page_title="Percnon Gibbesi Detection",
    page_icon="🦀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .detection-info {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1e88e5;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        from huggingface_hub import hf_hub_download

        with st.spinner("📥 Downloading model from Hugging Face..."):
            model_path = hf_hub_download(
                repo_id="roberalcaraz/percnon-detector",
                filename="percnon-detector-model.pt",
                cache_dir="./model_cache"
            )

            model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=model_path,
                confidence_threshold=0.5,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            st.success("✅ Model loaded successfully!")
            return model

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def detect_crabs(model, image):
    """Run crab detection with SAHI"""
    try:
        results = get_sliced_prediction(
            image,
            model,

            # tamaño del tile
            slice_height=1280,
            slice_width=1280,

            # solape
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        return results

    except Exception as e:
        st.error(f"Error during detection: {e}")
        return None

def draw_detections(image, results, conf_threshold=0.5):
    if results is None:
        return image, []

    annotated_image = image.copy()
    detections = []

    for pred in results.object_prediction_list:

        confidence = pred.score.value

        if confidence >= conf_threshold:

            x1 = int(pred.bbox.minx)
            y1 = int(pred.bbox.miny)
            x2 = int(pred.bbox.maxx)
            y2 = int(pred.bbox.maxy)

            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                10
            )

            label = f"Crab {confidence:.2f}"

            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,
                (255, 0, 0),
                10
            )

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "area": (x2 - x1) * (y2 - y1)
            })

    return annotated_image, detections


    """Draw bounding boxes on image"""
    if results is None:
        return image, []
        
    annotated_image = image.copy()
    detections = []
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                if box.conf.item() >= conf_threshold:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    confidence = box.conf.item()
                    
                    # Draw rectangle
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
                    
                    # Add label
                    label = f"Crab {confidence:.2f}"
                    cv2.putText(annotated_image, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 10)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'area': (x2-x1) * (y2-y1)
                    })
    
    return annotated_image, detections

def create_coco_annotation(detections, image_filename, image_width, image_height, image_id=1):
    """Create COCO format annotation for a single image"""
    
    # COCO format structure
    coco_data = {
        "images": [
            {
                "id": image_id,
                "file_name": image_filename,
                "width": image_width,
                "height": image_height
            }
        ],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "crab",
                "supercategory": "animal"
            }
        ]
    }
    
    # Add annotations for each detection
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['bbox']
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        annotation = {
            "id": idx + 1,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x1, y1, width, height],  # COCO format: [x, y, width, height]
            "area": area,
            "iscrowd": 0,
            "confidence": detection['confidence']
        }
        coco_data["annotations"].append(annotation)
    
    return coco_data

def make_batch_key(uploaded_files, conf_threshold):
    parts = [str(conf_threshold)]

    for f in uploaded_files:
        parts.append(f.name)
        parts.append(str(f.size))

    joined = "|".join(parts)

    return hashlib.md5(joined.encode()).hexdigest()

def create_yolo_annotation(detections, image_width, image_height):
    """Create YOLO format annotation (normalized coordinates)"""
    yolo_lines = []
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        
        # Convert to YOLO format (center_x, center_y, width, height) normalized
        center_x = ((x1 + x2) / 2) / image_width
        center_y = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        
        # Class ID 0 for crab, followed by normalized coordinates
        yolo_line = f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)
    
    return "\n".join(yolo_lines)

def create_detection_chart(detections):
    """Create confidence distribution chart"""
    if not detections:
        return None
    
    confidences = [d['confidence'] for d in detections]
    fig = px.histogram(
        x=confidences, 
        nbins=10,
        title="Detection Confidence Distribution",
        labels={'x': 'Confidence Score', 'y': 'Number of Detections'},
        color_discrete_sequence=['#1e88e5']
    )
    fig.update_layout(showlegend=False)
    return fig

def main():
    st.markdown('<h1 class="main-header">🦀 Percnon Gibbesi Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Detection Settings")
    
    # Model loading
    model = load_model()
    if model is None:
        st.error("❌ Model not loaded. Please check the configuration.")
        st.stop()
    
    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Lower values detect more crabs but may include false positives"
    )
    
    # Detection mode
    mode = st.sidebar.selectbox(
        "Detection Mode",
        ["Single Image", "Batch Processing", "Live Video"]
    )
    
    # Main content based on mode
    if mode == "Single Image":
        st.markdown('<h2 style="text-align: center;">📸 Single Image Detection</h2>', unsafe_allow_html=True)
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an underwater image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear underwater image for crab detection"
        )
        
        if uploaded_file is not None:
            # Load and display original image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, width="stretch")
            
            with col2:
                st.subheader("Detection Results")
                
                # Run detection
                with st.spinner("🔍 Detecting crabs..."):
                    results = detect_crabs(model, image_np)
                    annotated_image, detections = draw_detections(
                        image_np, results, conf_threshold
                    )
                
                st.image(annotated_image, width="stretch")
            
            # Results summary
            st.markdown("---")
            st.subheader("📊 Detection Summary")
            
            if detections:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Total Crabs</h3>
                        <h2>{len(detections)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    avg_conf = np.mean([d['confidence'] for d in detections])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Avg Confidence</h3>
                        <h2>{avg_conf:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    max_conf = max([d['confidence'] for d in detections])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Best Detection</h3>
                        <h2>{max_conf:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    total_area = sum([d['area'] for d in detections])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Total Area</h3>
                        <h2>{total_area:,} px²</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed results table
                st.subheader("🔍 Detailed Detections")
                df = pd.DataFrame(detections)
                df.index = df.index + 1
                df.index.name = "Detection #"
                st.dataframe(df, width="stretch")
                
                # Download options
                st.subheader("📥 Download Annotations")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Prepare image bytes for download (preserve original format)
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format=image.format or 'PNG')
                    img_buffer.seek(0)

                    st.download_button(
                        label="🖼️ Download Image",
                        data=img_buffer.getvalue(),
                        file_name=uploaded_file.name,
                        mime=uploaded_file.type if hasattr(uploaded_file, 'type') else 'image/png',
                        help="Download the original uploaded image (orientation normalized)"
                    )
                
                with col2:
                    # YOLO format
                    yolo_data = create_yolo_annotation(detections, image.width, image.height)
                    
                    st.download_button(
                        label="📄 Download YOLO TXT",
                        data=yolo_data,
                        file_name=f"{uploaded_file.name.split('.')[0]}.txt",
                        mime="text/plain",
                        help="YOLO format annotations"
                    )
                
                with col3:
                    # CSV format
                    df_download = pd.DataFrame(detections)
                    csv_data = df_download.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv_data,
                        file_name=f"{uploaded_file.name.split('.')[0]}_detections.csv",
                        mime="text/csv",
                        help="Detection data in CSV format"
                    )
                
            else:
                st.info("No crabs detected in this image. Try adjusting the confidence threshold.")
    
    elif mode == "Batch Processing":
        st.markdown('<h2 style="text-align: center;">📁 Batch Image Processing</h2>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )

        if uploaded_files:

            current_key = make_batch_key(
                uploaded_files,
                conf_threshold
            )

            if st.session_state.batch_key != current_key:

                st.write(f"Processing {len(uploaded_files)} images...")

                batch_results = []
                crab_results = []
                batch_annotations = {}
                prediction_images = {}

                progress_bar = st.progress(0)

                for i, uploaded_file in enumerate(uploaded_files):
                    image = Image.open(uploaded_file)
                    image_np = np.array(image)

                    image.save(f"{uploaded_file.name}")

                    results = detect_crabs(model, image_np)

                    annotated_image, detections = draw_detections(
                        image_np,
                        results,
                        conf_threshold
                    )

                    prediction_images[uploaded_file.name] = annotated_image

                    batch_results.append({
                        'filename': uploaded_file.name,
                        'crab_count': len(detections),
                        'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0,
                        'max_confidence': max([d['confidence'] for d in detections]) if detections else 0
                    })

                    for detection in detections:
                        x1, y1, x2, y2 = detection["bbox"]

                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        bbox_area = bbox_width * bbox_height

                        crab_results.append({
                            "filename": uploaded_file.name,
                            "confidence": detection["confidence"],
                            "bbox_area": bbox_area,
                            "bbox_width": bbox_width,
                            "bbox_height": bbox_height
                        })

                    batch_annotations[uploaded_file.name] = {
                        'detections': detections,
                        'width': image.width,
                        'height': image.height
                    }

                    progress_bar.progress((i + 1) / len(uploaded_files))

                st.session_state.images_df = pd.DataFrame(batch_results)
                st.session_state.crabs_df = pd.DataFrame(crab_results)
                st.session_state.batch_annotations = batch_annotations

                pred_zip_buffer = io.BytesIO()

                with zipfile.ZipFile(
                    pred_zip_buffer,
                    "w",
                    zipfile.ZIP_DEFLATED
                ) as zip_file:

                    for filename, img_array in prediction_images.items():

                        img_pil = Image.fromarray(img_array)

                        img_buffer = io.BytesIO()

                        ext = filename.split(".")[-1].upper()

                        if ext == "JPG":
                            ext = "JPEG"

                        img_pil.save(
                            img_buffer,
                            format=ext if ext in ["PNG", "JPEG"] else "PNG"
                        )

                        img_buffer.seek(0)

                        zip_file.writestr(
                            filename,
                            img_buffer.getvalue()
                        )

                pred_zip_buffer.seek(0)

                st.session_state.batch_predictions_zip = pred_zip_buffer.getvalue()

                st.session_state.batch_key = current_key

            st.subheader("📊 Batch Processing Results")

            summary_view = st.radio(
                "Choose summary",
                ["Images Summary", "Crabs Summary"],
                horizontal=True
            )

            if summary_view == "Images Summary":
                st.dataframe(
                    st.session_state.images_df,
                    width="stretch"
                )
            else:
                st.dataframe(
                    st.session_state.crabs_df,
                    width="stretch"
                )

            st.subheader("📥 Download CSV Results")

            col1, col2 = st.columns(2)

            images_csv = st.session_state.images_df.to_csv(index=False)
            crabs_csv = st.session_state.crabs_df.to_csv(index=False)

            with col1:
                st.download_button(
                    label="📊 Download Images Summary CSV",
                    data=images_csv,
                    file_name=f"images_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col2:
                st.download_button(
                    label="🦀 Download Crabs Summary CSV",
                    data=crabs_csv,
                    file_name=f"crabs_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            st.subheader("📥 Download Batch Annotations & Images")

            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(
                zip_buffer,
                'w',
                zipfile.ZIP_DEFLATED
            ) as zip_file:

                for filename, data in st.session_state.batch_annotations.items():

                    try:
                        matching = next(
                            (f for f in uploaded_files if f.name == filename),
                            None
                        )

                        if matching is not None:
                            try:
                                img = Image.open(filename)

                                out_buf = io.BytesIO()

                                img.save(
                                    out_buf,
                                    format=img.format or 'PNG'
                                )

                                out_buf.seek(0)

                                zip_file.writestr(
                                    filename,
                                    out_buf.getvalue()
                                )

                            except Exception:
                                zip_file.writestr(
                                    filename,
                                    matching.getvalue()
                                )

                    except Exception:
                        pass

                    if data["detections"]:

                        yolo_data = create_yolo_annotation(
                            data["detections"],
                            data["width"],
                            data["height"]
                        )

                        zip_file.writestr(
                            f"{filename.split('.')[0]}.txt",
                            yolo_data
                        )

            zip_buffer.seek(0)

            st.download_button(
                label="🗜️ Download Images + YOLO Annotations (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"batch_images_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                help="ZIP file containing original images and YOLO annotations for images with detections"
            )

            st.download_button(
                label="📦 Download Predictions (ZIP)",
                data=st.session_state.batch_predictions_zip,
                file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                help="ZIP with images including predicted bounding boxes"
            )
            
    
    elif mode == "Live Video":
        st.markdown('<h2 style="text-align: center;">🎥 Live Video Detection</h2>', unsafe_allow_html=True)

        st.info("🚧 Live video detection feature coming soon!")
        st.markdown("""
        **Features in development:**
        - Real-time webcam detection
        - RTSP stream processing
        - Video file upload and processing
        - Real-time statistics tracking
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        🦀 Percnon Gibbesi Detection System | Built with Streamlit & YOLOv12<br>
        <small>Model hosted on Hugging Face Hub</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
