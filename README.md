# 🏢 Smart Office Object Detection System

A comprehensive computer vision system for detecting key objects in office environments using YOLOv8. This system can accurately identify and locate people, chairs, monitors, keyboards, laptops, and smartphones in office CCTV footage or images.

## 🎯 Project Overview

The Smart Office challenge aims to build an intelligent monitoring system that can identify essential elements within modern office environments. This solution provides:

- **Object Detection**: Accurate bounding box detection and classification
- **Real-time Processing**: Fast inference for live monitoring
- **Interactive Dashboard**: Streamlit web interface for easy interaction
- **Comprehensive Evaluation**: Detailed performance metrics and analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 2GB+ free disk space

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd smart-office-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download datasets and train model**
```bash
python train.py
```

4. **Evaluate model performance**
```bash
python evaluate.py --model runs/detect/train2/weights/best.pt
```

5. **Launch the dashboard**
```bash
streamlit run dashboard.py
```

## 📊 Model Performance

Our trained YOLOv8 model achieves excellent performance on the Smart Office dataset:

| Metric | Value |
|--------|-------|
| **mAP50** | 0.977 (97.7%) |
| **mAP50-95** | 0.809 (80.9%) |
| **Precision** | 0.916 (91.6%) |
| **Recall** | 0.921 (92.1%) |
| **F1-Score** | 0.919 (91.9%) |

### Per-Class Performance

| Class | mAP50 | Precision | Recall |
|-------|-------|-----------|--------|
| Person | 0.931 | 0.991 | 0.784 |
| Chair | 0.977 | 0.891 | 0.820 |
| Monitor | 0.995 | 0.947 | 1.000 |
| Keyboard | 0.995 | 0.900 | 1.000 |
| Laptop | 0.995 | 0.947 | 1.000 |
| Phone | 0.990 | 0.852 | 1.000 |

## 🏗️ System Architecture

### Dataset
- **Source**: Roboflow datasets for office objects
- **Classes**: 6 object categories (person, chair, monitor, keyboard, laptop, phone)
- **Training**: 1,575 images
- **Validation**: 62 images  
- **Test**: 65 images

### Model
- **Architecture**: YOLOv8m (medium)
- **Input Size**: 640x640 pixels
- **Training**: 30 epochs
- **Optimizer**: Adam with cosine learning rate scheduling

### Features
- **Multi-class Detection**: Simultaneous detection of 6 office object types
- **Real-time Inference**: ~50ms inference time on GPU
- **Confidence Thresholding**: Adjustable detection sensitivity
- **Batch Processing**: Support for multiple image processing
- **Visualization**: Bounding boxes with class labels and confidence scores

## 📁 Project Structure

```
smart-office-detection/
├── train.py                 # Training script
├── evaluate.py              # Evaluation script  
├── dashboard.py             # Streamlit dashboard
├── requirements.txt         # Dependencies
├── README.md               # This file
├── smart_office_dataset/   # Dataset directory
│   ├── data.yaml          # Dataset configuration
│   ├── train/             # Training images and labels
│   ├── valid/             # Validation images and labels
│   └── test/              # Test images and labels
├── runs/detect/           # Training outputs
│   └── train2/           # Best trained model
│       └── weights/
│           └── best.pt   # Model weights
└── evaluation_results/    # Evaluation outputs
```

## 🔧 Usage

### Training

```bash
# Full training pipeline
python train.py

# Individual steps
python train.py --download    # Download datasets
python train.py --setup       # Setup dataset structure
python train.py --combine     # Combine datasets
python train.py --remap       # Remap class IDs
python train.py --train       # Train model

# Custom training parameters
python train.py --epochs 50 --batch 32 --imgsz 640
```

### Evaluation

```bash
# Comprehensive evaluation
python evaluate.py --model runs/detect/train2/weights/best.pt

# Custom evaluation parameters
python evaluate.py --model best.pt --conf 0.5 --iou 0.7 --output results/
```

### Dashboard

```bash
# Launch Streamlit dashboard
streamlit run dashboard.py
```

The dashboard provides:
- **Image Upload**: Drag and drop images for detection
- **Camera Input**: Real-time detection using webcam
- **Sample Images**: Test with provided sample images
- **Analytics**: Performance metrics and visualizations
- **Batch Processing**: Process multiple images at once

## 🎨 Dashboard Features

### Detection Interface
- **Multiple Input Methods**: Upload, camera, or sample images
- **Real-time Processing**: Instant detection results
- **Confidence Control**: Adjustable detection sensitivity
- **Visual Results**: Bounding boxes with labels and confidence scores

### Analytics Dashboard
- **Performance Metrics**: mAP, precision, recall, F1-score
- **Class Performance**: Per-class detection accuracy
- **Speed Metrics**: Inference time and FPS
- **Visual Charts**: Interactive performance visualizations

### Batch Processing
- **Multiple Images**: Process multiple files simultaneously
- **Progress Tracking**: Real-time progress indicators
- **Summary Reports**: Detailed batch processing results

## 📈 Performance Analysis

### Speed Performance
- **Average Inference Time**: ~50ms per image
- **FPS**: ~20 FPS on Tesla T4 GPU
- **Memory Usage**: ~2GB GPU memory
- **CPU Usage**: ~15% during inference

### Accuracy Analysis
- **High Precision**: 91.6% precision across all classes
- **Good Recall**: 92.1% recall for object detection
- **Balanced Performance**: Consistent performance across all object types
- **Robust Detection**: Works well in various office lighting conditions

## 🔍 Detection Examples

The system successfully detects:
- **People**: Office workers, visitors, staff
- **Furniture**: Office chairs, desk chairs
- **Electronics**: Computer monitors, keyboards, laptops, smartphones
- **Multiple Objects**: Simultaneous detection of multiple objects in complex scenes

## 🛠️ Technical Details

### Model Architecture
- **Backbone**: CSPDarknet53 with CSP connections
- **Neck**: PANet with feature pyramid network
- **Head**: YOLOv8 detection head with anchor-free design
- **Loss Function**: CIoU loss for bounding box regression

### Training Strategy
- **Data Augmentation**: Mosaic, mixup, random crops, color jittering
- **Learning Rate**: Cosine annealing with warmup
- **Optimizer**: AdamW with weight decay
- **Regularization**: Dropout, label smoothing

### Inference Pipeline
1. **Preprocessing**: Resize to 640x640, normalize
2. **Inference**: Forward pass through YOLOv8 model
3. **Post-processing**: NMS, confidence filtering
4. **Visualization**: Draw bounding boxes and labels

## 🚀 Deployment

### Local Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.port=8501"]
```

### Cloud Deployment
- **Heroku**: Deploy Streamlit app directly
- **AWS**: Use EC2 with GPU instances
- **Google Cloud**: Deploy on GCP with TPU/GPU
- **Azure**: Use Azure ML for model serving

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 implementation
- **Roboflow**: Dataset hosting and annotation tools
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision library

## 📞 Contact

For questions or support:
- **Email**: [your-email@example.com]
- **GitHub**: [your-github-username]
- **LinkedIn**: [your-linkedin-profile]

## 📚 References

1. Jocher, G., et al. "YOLOv8 by Ultralytics." GitHub repository, 2023.
2. Roboflow. "Computer Vision Dataset Management Platform." 2023.
3. Streamlit. "The fastest way to build and share data apps." 2023.

---

**Built with ❤️ for the Smart Office Challenge**#   Y o l o v 8 T r a i n  
 