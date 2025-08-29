# Self-Driving Car Computer Vision System - Setup Guide

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 20.04+ / Windows 10+ / macOS 10.15+
- **Python**: 3.8 - 3.11
- **RAM**: Minimum 8GB, Recommended 16GB+
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 10GB+ free space

### Hardware Requirements (Optional)
- **Camera**: USB webcam or multiple cameras for 360° vision
- **Processing Unit**: NVIDIA Jetson (for embedded deployment)
- **Sensors**: LiDAR, ultrasonic sensors (for advanced features)

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/self-driving-car-cv.git
cd self-driving-car-cv
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv self_driving_env
source self_driving_env/bin/activate  # On Windows: self_driving_env\Scripts\activate

# Or using conda
conda create -n self_driving_env python=3.9
conda activate self_driving_env
```

### 3. Install Dependencies

#### Basic Installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### GPU Support (NVIDIA CUDA)
```bash
# First, install CUDA toolkit from NVIDIA website
# Then install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Advanced Computer Vision Libraries
```bash
# Detectron2 (Facebook's object detection)
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# MediaPipe (Google's ML solutions)
pip install mediapipe

# YOLO v8 (Ultralytics)
pip install ultralytics
```

### 4. Verify Installation
```python
# Run this test script
python -c "
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

print('OpenCV version:', cv2.__version__)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
print('NumPy version:', np.__version__)
print('✓ All core dependencies installed successfully!')
"
```

## Configuration

### 1. Camera Setup
```python
# Test camera connection
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print('✓ Camera working')
    print('Resolution:', frame.shape)
else:
    print('✗ Camera not detected')
cap.release()
"
```

### 2. Model Downloads
```bash
# Download pre-trained models
mkdir models
cd models

# YOLO weights (auto-downloaded on first run)
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')"

# Or download manually:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 3. Configuration Files
Create `config.yaml`:
```yaml
# Camera Configuration
cameras:
  front:
    device_id: 0
    resolution: [1920, 1080]
    fps: 30
  
# Model Configuration
models:
  object_detection: "yolov8n.pt"
  lane_detection: "custom"
  
# System Configuration
system:
  max_speed: 50  # km/h
  safety_distance: 10  # meters
  confidence_threshold: 0.5
```

## Running the System

### 1. Quick Demo
```bash
# Run simulation mode (no hardware required)
python main.py --mode simulation

# Run with real camera
python main.py --mode camera --camera-id 0
```

### 2. Full System
```bash
# Complete autonomous system
python autonomous_vehicle.py --config config.yaml
```

### 3. Individual Components
```bash
# Test lane detection only
python test_lane_detection.py

# Test object detection only
python test_object_detection.py

# Test traffic sign detection
python test_traffic_signs.py
```

## Troubleshooting

### Common Issues

#### 1. OpenCV Camera Issues
```bash
# Linux: Camera permissions
sudo usermod -a -G video $USER
# Log out and log back in

# Windows: Install camera drivers
# Check Windows Device Manager
```

#### 2. CUDA/GPU Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Memory Issues
```python
# Reduce batch size in config
batch_size = 1  # Instead of 32

# Enable memory optimization
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
```

#### 4. Import Errors
```bash
# Missing system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# macOS
brew install libomp
```

## Performance Optimization

### 1. For Real-time Processing
```python
# Use smaller models
model = YOLO('yolov8n.pt')  # nano version

# Reduce input resolution
input_size = (416, 416)  # Instead of (640, 640)

# Enable TensorRT (NVIDIA GPUs)
model.export(format='engine')
```

### 2. For Embedded Systems (Raspberry Pi/Jetson)
```bash
# Install optimized OpenCV
pip install opencv-python-headless

# Use quantized models
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', int8=True)
"
```

## Development Setup

### 1. Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### 2. Code Quality Tools
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### 3. Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/
```

## Deployment Options

### 1. Edge Deployment (NVIDIA Jetson)
```bash
# Install JetPack SDK
sudo apt install nvidia-jetpack

# Optimize for inference
python optimize_for_jetson.py
```

### 2. Cloud Deployment
```bash
# Docker container
docker build -t self-driving-car .
docker run --gpus all -p 8000:8000 self-driving-car
```

### 3. Web Interface
```bash
# Start web dashboard
streamlit run dashboard.py --server.port 8501
```

## Hardware Integration

### 1. Arduino/Microcontroller
```python
# Install serial communication
pip install pyserial

# Test connection
python test_hardware_interface.py
```

### 2. CAN Bus Integration
```bash
pip install python-can
```

## Additional Resources

- **Documentation**: `/docs` folder
- **Examples**: `/examples` folder  
- **Notebooks**: `/notebooks` for Jupyter tutorials
- **Models**: `/models` for pre-trained weights
- **Config**: `/config` for configuration files

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with system info and error logs
4. Join our Discord community for real-time support

## License

This project is licensed under the MIT License - see LICENSE file for details.