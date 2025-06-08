# hybrid_document_classifier.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
import pytesseract
from tqdm import tqdm

# ======================
# CONFIG
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_NAME = "bert-base-uncased"
IMAGE_SIZE = (224, 224)

# ==========================
# DATASET CLASS
# ==========================
class DocumentDataset(Dataset):
    def __init__(self, root_dir, tokenizer, transform):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(image)

        if self.transform:
            image = self.transform(image)

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": encoded['input_ids'].squeeze(0),
            "attention_mask": encoded['attention_mask'].squeeze(0),
            "label": torch.tensor(label)
        }

# ==========================
# HYBRID MODEL
# ==========================
class HybridDocumentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(HybridDocumentClassifier, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.cnn_out = 512

        self.bert = BertModel.from_pretrained(BERT_NAME)
        self.bert_out = 768

        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_out + self.bert_out, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.cnn(image)
        text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        combined = torch.cat((img_feat, text_feat), dim=1)
        return self.classifier(combined)

# ==========================
# TRAINING FUNCTION
# ==========================
def train(model, dataloader, optimizer, criterion, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        total_loss, total_correct, total_samples = 0, 0, 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images = batch["image"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

        print(f"Epoch {epoch+1}, Loss: {total_loss / total_samples:.4f}, Accuracy: {100 * total_correct / total_samples:.2f}%")

# ==========================
# MAIN SCRIPT
# ==========================
if __name__ == "__main__":
    data_dir = "dataset_split/train_df"
    tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = DocumentDataset(data_dir, tokenizer, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = HybridDocumentClassifier(num_classes=len(dataset.classes)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    train(model, dataloader, optimizer, criterion, num_epochs=3)

    torch.save(model.state_dict(), "hybrid_doc_classifier.pth")
