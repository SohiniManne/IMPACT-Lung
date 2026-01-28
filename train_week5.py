import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# Imports
from src.config import BATCH_SIZE
from src.imaging_loader import get_imaging_loader
from src.sensor_loader import get_sensor_loader
from src.emr_loader import get_emr_loader
from src.attention_model import AttentionFusionModel # <--- NEW MODEL

# --- CONFIGURATION ---
NUM_EPOCHS = 5
LEARNING_RATE = 0.0005 # Lower LR for Attention models usually helps
CHECKPOINT_DIR = './checkpoints'

def train_attention_system():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting WEEK 5 Training (Attention Fusion) on: {device}")

    # 1. Load Data
    loader_img = get_imaging_loader(split='train')
    loader_sensor = get_sensor_loader()
    loader_emr = get_emr_loader()
    min_len = min(len(loader_img), len(loader_sensor), len(loader_emr))

    # 2. Initialize Attention Model
    model = AttentionFusionModel(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()
    
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        iterator = zip(loader_img, loader_sensor, loader_emr)
        
        for step, (batch_img, batch_sensor, batch_emr) in enumerate(iterator):
            # Move to device
            imgs = batch_img[0].to(device)
            sensors = batch_sensor[0].to(device)
            labels = batch_sensor[1].to(device)
            emr = batch_emr[0].to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(imgs, sensors, emr)
            
            # Loss & Backward
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        avg_loss = running_loss / min_len
        print(f"✨ Epoch {epoch+1}: Loss={avg_loss:.4f} | Acc={acc:.2f}%")
        
        # Save Checkpoint
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'impact_lung_attention.pth'))

    print("✅ Week 5 Training Complete!")

if __name__ == "__main__":
    train_attention_system()