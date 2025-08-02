import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import tempfile
from pathlib import Path
import json
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="Smart Office Object Detection",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .detection-box {
        border: 2px solid #ff6b6b;
        border-radius: 4px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        background-color: #fff5f5;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class SmartOfficeDetector:
    """Smart Office Object Detection Class"""
    
    def __init__(self, model_path="runs/detect/train2/weights/best.pt"):
        self.model_path = model_path
        self.model = None
        self.class_names = ['person', 'chair', 'monitor', 'keyboard', 'laptop', 'phone']
        self.colors = {
            'person': '#FF6B6B',
            'chair': '#4ECDC4', 
            'monitor': '#45B7D1',
            'keyboard': '#96CEB4',
            'laptop': '#FFEAA7',
            'phone': '#DDA0DD'
        }
        self.load_model()
    
    def load_model(self):
        """Load the YOLOv8 model"""
        try:
            # Check if ultralytics is installed
            try:
                from ultralytics import YOLO
            except ImportError:
                st.error("❌ Ultralytics module not found!")
                st.info("Please install ultralytics: `pip install ultralytics`")
                return
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                st.warning(f"⚠️ Model file not found: {self.model_path}")
                st.info("Please run training first: `python train.py --train`")
                return
            
            # Load the model
            self.model = YOLO(self.model_path)
            st.success("✅ Model loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.info("Please ensure the model file exists and ultralytics is installed.")
    
    def detect_objects(self, image, conf_threshold=0.5):
        """Perform object detection on image"""
        if self.model is None:
            return None, "Model not loaded"
        
        try:
            # Run inference
            results = self.model.predict(
                image, 
                conf=conf_threshold,
                verbose=False
            )
            
            return results[0], None
        except Exception as e:
            return None, str(e)
    
    def draw_detections(self, image, results):
        """Draw detection boxes on image"""
        if results is None:
            return image
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Assume image is in RGB or grayscale, not BGR (since no cv2)
            if image.ndim == 2:  # grayscale
                pil_image = Image.fromarray(image)
            elif image.shape[2] == 3:  # RGB
                pil_image = Image.fromarray(image)
            elif image.shape[2] == 4:  # RGBA
                pil_image = Image.fromarray(image, mode="RGBA")
            else:
                pil_image = Image.fromarray(image)
        else:
            pil_image = image.copy()
        
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        detections = []
        
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes
            if len(boxes) > 0:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class and confidence
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    if cls < len(self.class_names):
                        class_name = self.class_names[cls]
                        color = self.colors.get(class_name, '#FF0000')
                        
                        # Draw bounding box
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        
                        # Draw label
                        label = f"{class_name} {conf:.2f}"
                        bbox = draw.textbbox((x1, y1-20), label, font=font)
                        draw.rectangle(bbox, fill=color)
                        draw.text((x1, y1-20), label, fill='white', font=font)
                        
                        detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2]
                        })
        
        return pil_image, detections

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏢 Smart Office Object Detection</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
        # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_path = st.selectbox(
            "Select Model",
            ["runs/detect/train2/weights/best.pt", "yolov8m.pt"],
            help="Choose the model to use for detection"
        )
        
        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Minimum confidence score for detections"
        )
        
        # Performance metrics
        st.header("📊 Performance")
        if os.path.exists("evaluation_results/comprehensive_evaluation.json"):
            with open("evaluation_results/comprehensive_evaluation.json", 'r') as f:
                eval_data = json.load(f)
            
            metrics = eval_data.get('evaluation_report', {}).get('performance_metrics', {})
            speed_metrics = eval_data.get('speed_metrics', {})
            
            st.metric("mAP50", f"{metrics.get('mAP50', 0):.3f}")
            st.metric("mAP50-95", f"{metrics.get('mAP50-95', 0):.3f}")
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            st.metric("FPS", f"{speed_metrics.get('fps', 0):.1f}")
        
        # Setup instructions
        st.header("🔧 Setup Instructions")
        # Note: detector will be initialized after sidebar
        st.info("""
        **To get started:**
        1. Install dependencies: `pip install -r requirements.txt`
        2. Train model: `python train.py --train`
        3. Refresh this page
        """)
        
        # About section
        st.header("ℹ️ About")
        st.info("""
        This system detects 6 key office objects:
        - 👤 Person
        - 🪑 Chair  
        - 🖥️ Monitor
        - ⌨️ Keyboard
        - 💻 Laptop
        - 📱 Phone
        """)
    
    # Initialize detector with selected model
    detector = SmartOfficeDetector(model_path)
    
    # Update setup instructions based on model status
    with st.sidebar:
        if detector.model is None:
            st.error("Model not loaded!")
        else:
            st.success("✅ Model loaded successfully!")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Detection", "📊 Analytics", "📁 Batch Processing", "ℹ️ Info"])
    
    with tab1:
        st.header("Object Detection")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload Image", "Camera", "Sample Images"]
        )
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image to detect objects"
            )
            
            if uploaded_file is not None:
                # Load image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Detection button
                if st.button("🔍 Detect Objects", type="primary"):
                    if detector.model is None:
                        st.error("❌ Model not loaded. Please check the sidebar for setup instructions.")
                    else:
                        with st.spinner("Detecting objects..."):
                            start_time = time.time()
                            results, error = detector.detect_objects(image, conf_threshold)
                            inference_time = time.time() - start_time
                            
                            if error:
                                st.error(f"Detection error: {error}")
                            else:
                                # Draw detections
                                annotated_image, detections = detector.draw_detections(image, results)
                                
                                # Display results
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.image(annotated_image, caption="Detection Results", use_container_width=True)
                                
                                with col2:
                                    st.subheader("📋 Detection Results")
                                    st.metric("Inference Time", f"{inference_time:.3f}s")
                                    st.metric("Objects Detected", len(detections))
                                    
                                    if detections:
                                        st.subheader("Detected Objects:")
                                        for i, detection in enumerate(detections):
                                            with st.container():
                                                st.markdown(f"""
                                                <div class="detection-box">
                                                    <strong>{detection['class'].title()}</strong><br>
                                                    Confidence: {detection['confidence']:.3f}
                                                </div>
                                                """, unsafe_allow_html=True)
                                    else:
                                        st.info("No objects detected with current confidence threshold.")
        
        elif input_method == "Camera":
            camera_input = st.camera_input("Take a picture")
            
            if camera_input is not None:
                image = Image.open(camera_input)
                
                if st.button("🔍 Detect Objects", type="primary"):
                    if detector.model is None:
                        st.error("❌ Model not loaded. Please check the sidebar for setup instructions.")
                    else:
                        with st.spinner("Detecting objects..."):
                            results, error = detector.detect_objects(image, conf_threshold)
                            
                            if error:
                                st.error(f"Detection error: {error}")
                            else:
                                annotated_image, detections = detector.draw_detections(image, results)
                                st.image(annotated_image, caption="Detection Results", use_container_width=True)
                                
                                # Show detection summary
                                if detections:
                                    st.subheader("Detection Summary:")
                                    for detection in detections:
                                        st.write(f"• {detection['class'].title()}: {detection['confidence']:.3f}")
        
        elif input_method == "Sample Images":
            st.subheader("Sample Images")
            
            # Sample images from the dataset
            sample_images = ["1.png", "2.png"]
            
            selected_sample = st.selectbox("Choose a sample image:", sample_images)
            
            if selected_sample and os.path.exists(selected_sample):
                image = Image.open(selected_sample)
                st.image(image, caption=f"Sample: {selected_sample}", use_column_width=True)
                
                if st.button("🔍 Detect Objects", type="primary"):
                    if detector.model is None:
                        st.error("❌ Model not loaded. Please check the sidebar for setup instructions.")
                    else:
                        with st.spinner("Detecting objects..."):
                            results, error = detector.detect_objects(image, conf_threshold)
                            
                            if error:
                                st.error(f"Detection error: {error}")
                            else:
                                annotated_image, detections = detector.draw_detections(image, results)
                                st.image(annotated_image, caption="Detection Results", use_container_width=True)
    
    with tab2:
        st.header("Analytics Dashboard")
        
        if os.path.exists("evaluation_results/comprehensive_evaluation.json"):
            with open("evaluation_results/comprehensive_evaluation.json", 'r') as f:
                eval_data = json.load(f)
            
            # Performance metrics
            metrics = eval_data.get('evaluation_report', {}).get('performance_metrics', {})
            class_performance = eval_data.get('evaluation_report', {}).get('class_performance', {})
            
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("mAP50", f"{metrics.get('mAP50', 0):.3f}")
            with col2:
                st.metric("mAP50-95", f"{metrics.get('mAP50-95', 0):.3f}")
            with col3:
                st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            with col4:
                st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            
            # Class performance chart
            if class_performance:
                st.subheader("Class Performance")
                
                # Create DataFrame for plotting
                class_data = []
                for class_name, perf in class_performance.items():
                    class_data.append({
                        'Class': class_name,
                        'mAP50': perf.get('mAP50', 0),
                        'Precision': perf.get('precision', 0),
                        'Recall': perf.get('recall', 0)
                    })
                
                df = pd.DataFrame(class_data)
                
                # Bar chart
                fig = px.bar(df, x='Class', y='mAP50', 
                           title='mAP50 Scores by Class',
                           color='mAP50',
                           color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.subheader("Detailed Class Performance")
                st.dataframe(df, use_container_width=True)
        
        else:
            st.info("No evaluation results found. Run the evaluation script first.")
    
    with tab3:
        st.header("Batch Processing")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images for batch processing",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")
            
            if st.button("🔍 Process All Images", type="primary"):
                if detector.model is None:
                    st.error("❌ Model not loaded. Please check the sidebar for setup instructions.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results_summary = []
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing image {i+1}/{len(uploaded_files)}")
                        
                        image = Image.open(uploaded_file)
                        results, error = detector.detect_objects(image, conf_threshold)
                        
                        if not error:
                            _, detections = detector.draw_detections(image, results)
                            
                            # Count detections by class
                            class_counts = {}
                            for detection in detections:
                                class_name = detection['class']
                                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                            
                            results_summary.append({
                                'filename': uploaded_file.name,
                                'total_detections': len(detections),
                                'class_counts': class_counts
                            })
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display batch results
                st.subheader("Batch Processing Results")
                
                if results_summary:
                    # Summary table
                    summary_df = pd.DataFrame(results_summary)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Overall statistics
                    total_detections = sum(r['total_detections'] for r in results_summary)
                    st.metric("Total Detections", total_detections)
                    st.metric("Average Detections per Image", f"{total_detections/len(results_summary):.1f}")
    
    with tab4:
        st.header("System Information")
        
        # Model information
        st.subheader("Model Details")
        st.write(f"**Model Path:** {model_path}")
        st.write(f"**Classes:** {', '.join(detector.class_names)}")
        st.write(f"**Confidence Threshold:** {conf_threshold}")
        
        # System requirements
        st.subheader("System Requirements")
        st.write("""
        - Python 3.8+
        - CUDA-compatible GPU (recommended)
        - 8GB+ RAM
        - 2GB+ free disk space
        """)
        
        # Installation instructions
        st.subheader("Installation")
        st.code("""
pip install -r requirements.txt
python train.py --train
python evaluate.py --model runs/detect/train2/weights/best.pt
streamlit run dashboard.py
        """)
        
        # Usage instructions
        st.subheader("Usage")
        st.write("""
        1. **Training:** Run `python train.py` to train the model
        2. **Evaluation:** Run `python evaluate.py` to evaluate performance
        3. **Dashboard:** Run `streamlit run dashboard.py` to start the web interface
        4. **Detection:** Upload images or use camera for real-time detection
        """)

if __name__ == "__main__":
    main() 