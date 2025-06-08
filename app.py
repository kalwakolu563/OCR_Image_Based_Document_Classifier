import os
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import cv2
import numpy as np

# Fix for duplicate symbols error in PyTorch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Optional workaround for Streamlit introspection issue with torch.classes
try:
    import sys
    if 'torch.classes' in sys.modules:
        del sys.modules['torch.classes']
except Exception:
    pass

# Set title
st.title("📄 OCR Document Classifier")
st.markdown("Upload a document image to predict its class (e.g., invoice, receipt, etc.)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Get class names from train folder
train_dir = "dataset_split/train_df"
class_names = sorted(os.listdir(train_dir))

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load("ocr_document_classifier.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Preprocess image
def preprocess_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        img_cv = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img_cv is None:
            raise ValueError("Unreadable image file.")
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)

    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dim
    return image.to(device)

# Predict function
def predict(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Upload and classify
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    try:
        image_tensor = preprocess_image(uploaded_file)
        prediction = predict(image_tensor)
        predicted_class_name = class_names[prediction]

        st.image(uploaded_file, caption="Uploaded Document", use_container_width=True)
        st.success(f"📌 Prediction: **{predicted_class_name}** (Class {prediction})")
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
