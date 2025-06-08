import os
import json
import pandas as pd
from PIL import Image, UnidentifiedImageError
import pytesseract
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# DEVICE SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PARAMETERS
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_EPOCHS = 2

# 1. IMAGE TRANSFORMS
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 2. TEXT EXTRACTION FROM IMAGES & LABEL MAPPING
def extract_text_from_images(root_dir, output_csv, label_map_path):
    data = []
    label_map = {}
    label_counter = 0

    for class_label in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_label)
        if not os.path.isdir(class_path):
            continue

        if class_label not in label_map:
            label_map[class_label] = label_counter
            label_counter += 1

        for filename in os.listdir(class_path):
            filepath = os.path.join(class_path, filename)
            try:
                image = Image.open(filepath).convert('RGB')
                text = pytesseract.image_to_string(image).strip()
                if not text:
                    continue  # skip images with no extracted text
                data.append({
                    "filename": filepath,
                    "text": text,
                    "label": label_map[class_label]
                })
            except (UnidentifiedImageError, OSError):
                print(f"Skipped corrupted file: {filepath}")

    df = pd.DataFrame(data)
    df.dropna(subset=["text"], inplace=True)
    df.to_csv(output_csv, index=False)

    with open(label_map_path, "w") as f:
        json.dump(label_map, f)

    return label_map

# Run this once to extract OCR text and generate cleaned CSVs
train_label_map = extract_text_from_images("dataset_split/train_df", "train_texts.csv", "train_label_map.json")
test_label_map = extract_text_from_images("dataset_split/test_df", "test_texts.csv", "test_label_map.json")

# 3. CUSTOM DATASET
class HybridDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image = Image.open(row['filename']).convert("RGB")
        image = self.transform(image)

        text = str(row['text']) if pd.notna(row['text']) else ""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        label = int(row['label'])

        return image, input_ids, attention_mask, label

# 4. HYBRID MODEL
class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridModel, self).__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.cnn(image)
        text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        combined = torch.cat((img_feat, text_feat), dim=1)
        return self.classifier(combined)

# 5. TRAIN AND EVALUATION FUNCTION
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for image, input_ids, attn_mask, label in tqdm(dataloader, desc="Training"):
        image, input_ids, attn_mask, label = image.to(device), input_ids.to(device), attn_mask.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image, input_ids, attn_mask)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(1) == label).sum().item()
    acc = 100 * correct / len(dataloader.dataset)
    print(f"Train Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for image, input_ids, attn_mask, label in tqdm(dataloader, desc="Evaluating"):
            image, input_ids, attn_mask, label = image.to(device), input_ids.to(device), attn_mask.to(device), label.to(device)
            output = model(image, input_ids, attn_mask)
            correct += (output.argmax(1) == label).sum().item()
    acc = 100 * correct / len(dataloader.dataset)
    print(f"Test Accuracy: {acc:.2f}%")

# 6. LOAD DATA
num_classes = len(train_label_map)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = HybridDataset("train_texts.csv", tokenizer, transform)
test_dataset = HybridDataset("test_texts.csv", tokenizer, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 7. INIT MODEL
model = HybridModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# 8. TRAINING LOOP
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    train_epoch(model, train_loader, optimizer, criterion)
    evaluate_model(model, test_loader)

# 9. SAVE MODEL
torch.save(model.state_dict(), "hybrid_model.pth")
print("✅ Model saved to hybrid_model.pth")
