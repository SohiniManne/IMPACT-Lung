import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.config import BATCH_SIZE
from src.imaging_loader import get_imaging_loader
from src.sensor_loader import get_sensor_loader
from src.emr_loader import get_emr_loader
from src.attention_model import AttentionFusionModel

# --- CONFIGURATION ---
NUM_EPOCHS = 5
LEARNING_RATE = 0.0001 # Fine-tuned: Lower LR for stability
CHECKPOINT_DIR = './checkpoints'

def train_weighted_optimization():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Week 6: Clinical Optimization (Weighted Loss) on {device}")

    # 1. Load Data
    loader_img = get_imaging_loader(split='train')
    loader_sensor = get_sensor_loader()
    loader_emr = get_emr_loader()
    min_len = min(len(loader_img), len(loader_sensor), len(loader_emr))

    # 2. Define Clinical Weights
    # "Normal" (Class 0) = 1.0
    # "Pathology" (Class 1) = 3.0  <-- PENALIZE MISSING A SICK PATIENT 3x MORE
    class_weights = torch.tensor([1.0, 3.0]).to(device)
    
    # 3. Initialize Model & Optimized Loss
    model = AttentionFusionModel(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Apply weights here
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model.train()
    
    print("   Training with Weighted Cross-Entropy...")
    
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        iterator = zip(loader_img, loader_sensor, loader_emr)
        
        for step, (batch_img, batch_sensor, batch_emr) in enumerate(iterator):
            # Move data
            imgs = batch_img[0].to(device)
            sensors = batch_sensor[0].to(device)
            labels = batch_sensor[1].to(device)
            emr = batch_emr[0].to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, sensors, emr)
            
            loss = criterion(outputs, labels) # Loss is now weighted
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"✨ Epoch {epoch+1}: Loss={running_loss/min_len:.4f} | Acc={100*correct/total:.2f}%")

    # Save the "Clinically Optimized" Model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'impact_lung_optimized.pth'))
    print("✅ Optimized Model Saved.")

if __name__ == "__main__":
    train_weighted_optimization()