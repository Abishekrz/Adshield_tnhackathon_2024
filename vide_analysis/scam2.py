import cv2
import os
import pytesseract
import torch
import whisper
import moviepy.editor as mp
import speech_recognition as sr
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from PIL import Image

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Scam-related keywords
SCAM_KEYWORDS = [
    "free money", "click this link", "urgent", "congratulations", 
    "lottery", "giveaway", "limited offer", "investment", "crypto scheme",
    "xbet", "win big", "2x", "2x profit"
]

# Load pre-trained ResNet50 and modify classifier
resnet = models.resnet50(pretrained=True)
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 2)  # Scam (1) or Safe (0)
resnet.to(device)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load Whisper model
whisper_model = whisper.load_model("base").to(device)

# Load DistilBERT model for scam text classification
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
distilbert.to(device)

# Function to extract frames from video
def extract_frames(video_path, output_folder="frames"):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    return output_folder

# Custom dataset class for training image classifier
class ScamDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        for label, category in enumerate(["safe", "scam"]):
            category_dir = os.path.join(root_dir, category)
            for img_name in os.listdir(category_dir):
                self.samples.append((os.path.join(category_dir, img_name), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Train image classifier
def train_resnet_classifier():
    dataset = ScamDataset(root_dir="dataset", transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=0.0001)
    
    num_epochs = 5
    for epoch in range(num_epochs):
        resnet.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")
    
    torch.save(resnet.state_dict(), "scam_detector.pth")

# Train text classifier
def train_text_classifier():
    dataset = {"train": "scam_text_data.csv"}
    data = []
    
    with open(dataset["train"], "r", encoding="utf-8") as f:
        for line in f.readlines():
            text, label = line.strip().split(",")
            data.append({"text": text, "label": int(label)})
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
    )
    
    trainer = Trainer(
        model=distilbert,
        args=training_args,
        train_dataset=data,
    )
    trainer.train()
    distilbert.save_pretrained("scam_text_model")

# Train both models
train_resnet_classifier()
train_text_classifier()
