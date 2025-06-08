# STEP 1: IMPORT LIBRARIES
import os
import pandas as pd
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np

# STEP 2: CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = (224, 224)
batch_size = 32
num_epochs = 2

# STEP 3: IMAGE TRANSFORMS
image_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# STEP 4: LOAD TEXT DATA
train_texts = pd.read_csv("train_texts.csv")  # columns: filename, text, label
test_texts = pd.read_csv("test_texts.csv")

# STEP 5: BERT EMBEDDINGS
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

train_embeddings = bert_model.encode(train_texts['text'].tolist(), convert_to_tensor=True)
test_embeddings = bert_model.encode(test_texts['text'].tolist(), convert_to_tensor=True)

# STEP 6: LABEL ENCODING
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_texts['label'])
y_test = label_encoder.transform(test_texts['label'])

# STEP 7: CUSTOM DATASET
class HybridDocumentDataset(Dataset):
    def __init__(self, df, text_embeddings, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.text_embeddings = text_embeddings
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        text_feat = self.text_embeddings[idx]
        label = torch.tensor(row['label'], dtype=torch.long)
        return image, text_feat, label

# STEP 8: DATASET AND DATALOADER
train_dataset = HybridDocumentDataset(train_texts.assign(label=y_train), train_embeddings, "dataset_split/train_df", image_transform)
test_dataset = HybridDocumentDataset(test_texts.assign(label=y_test), test_embeddings, "dataset_split/test_df", image_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# STEP 9: HYBRID MODEL (CNN + BERT)
class HybridClassifier(nn.Module):
    def __init__(self, num_classes, text_feat_dim):
        super(HybridClassifier, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False  # Freeze CNN
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.text_branch = nn.Sequential(
            nn.Linear(text_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + 256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, text_feat):
        img_feat = self.cnn(image)
        text_feat = self.text_branch(text_feat)
        combined = torch.cat((img_feat, text_feat), dim=1)
        return self.classifier(combined)

# STEP 10: INITIALIZE MODEL
text_embedding_size = train_embeddings.shape[1]  # Typically 384 for MiniLM
model = HybridClassifier(num_classes=len(label_encoder.classes_), text_feat_dim=text_embedding_size)
model = model.to(device)

# STEP 11: LOSS & OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# STEP 12: TRAINING FUNCTION
def train_model():
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, text_feats, labels in tqdm(train_loader):
            images, text_feats, labels = images.to(device), text_feats.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, text_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / total:.4f}, Accuracy: {100 * correct / total:.2f}%")
        evaluate_model()

# STEP 13: EVALUATION FUNCTION
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, text_feats, labels in test_loader:
            images, text_feats, labels = images.to(device), text_feats.to(device), labels.to(device)
            outputs = model(images, text_feats)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# STEP 14: RUN TRAINING
train_model()

# STEP 15: SAVE MODEL
torch.save(model.state_dict(), "hybrid_ocr_bert_classifier.pth")
