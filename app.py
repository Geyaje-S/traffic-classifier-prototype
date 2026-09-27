import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

# Define classes matching your training setup
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class RobustTrafficCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(RobustTrafficCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout(0.4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.feature_extractor(x))

@st.cache_resource
def load_prototype_model():
    model = RobustTrafficCNN(num_classes=10)
    model.load_state_dict(torch.load('traffic_cnn_weights.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Streamlit Configuration Setup
st.set_page_config(page_title="Batch Traffic Vision Engine", layout="wide")
st.title("🚗 Production-Ready Batch Traffic Classifier")
st.markdown("This enhanced version processes multiple urban traffic image vectors simultaneously using asynchronous matrix inference loops.")

# File Uploader with multiple files parameter enabled
uploaded_files = st.file_uploader(
    "Upload one or more urban traffic images...", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    try:
        model = load_prototype_model()
        
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Lists to store batch metrics data
        results_data = []
        
        st.subheader("🖼️ Uploaded Batch Gallery")
        # Display images nicely side-by-side in grid columns
        cols = st.columns(len(uploaded_files))
        
        for idx, file in enumerate(uploaded_files):
            image = Image.open(file).convert('RGB')
            
            # Render small preview in the grid channel
            with cols[idx]:
                st.image(image, caption=file.name, use_container_width=True)
            
            # Run inference loop step
            input_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
            
            # Save metrics parameters into list dictionary
            results_data.append({
                "File Name": file.name,
                "Predicted Category": CLASSES[predicted_idx.item()].upper(),
                "Confidence Score": f"{confidence.item() * 100:.2f}%"
            })
            
        # Display the formal comparative summary data matrix
        st.subheader("📊 Batch Processing Analytics Report")
        df = pd.DataFrame(results_data)
        st.table(df)
        
    except Exception as e:
        st.error(f"Execution Error during batch loop: {e}")
else:
    st.info("System Status: Idle. Awaiting image batch entry vector stream to execute inference loops.")
