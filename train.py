import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# Import our custom modules
from src.config import BATCH_SIZE
from src.imaging_loader import get_imaging_loader
from src.sensor_loader import get_sensor_loader
from src.emr_loader import get_emr_loader
from src.fusion_model import ImpactLungModel

# --- CONFIGURATION ---
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
CHECKPOINT_DIR = './checkpoints'

def train_impact_lung():
    # 1. Setup Environment
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Training on Device: {device}")
    print(f"   Batch Size: {BATCH_SIZE} | Epochs: {NUM_EPOCHS}")

    # 2. Load Data
    print("\n[1/4] Loading Datasets...")
    loader_img = get_imaging_loader(split='train') # Use 'train' for augmentation
    loader_sensor = get_sensor_loader()
    loader_emr = get_emr_loader()
    
    # Calculate steps per epoch based on the smallest dataset
    min_len = min(len(loader_img), len(loader_sensor), len(loader_emr))
    print(f"   Training steps per epoch: {min_len}")

    # 3. Initialize Model
    print("\n[2/4] Initializing IMPACT-Lung Model...")
    model = ImpactLungModel(num_classes=2).to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    print("\n[3/4] Beginning Training Loop...")
    model.train() # Set mode to training
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # We zip the loaders to iterate them simultaneously
        iterator = zip(loader_img, loader_sensor, loader_emr)
        
        for step, (batch_img, batch_sensor, batch_emr) in enumerate(iterator):
            
            # A. Unpack Data & Move to Device
            imgs = batch_img[0].to(device)
            sensors = batch_sensor[0].to(device)
            sensor_labels = batch_sensor[1].to(device) # Using sensor labels as ground truth
            emr = batch_emr[0].to(device)
            
            # B. Forward Pass
            optimizer.zero_grad()
            outputs = model(imgs, sensors, emr)
            
            # C. Calculate Loss
            loss = criterion(outputs, sensor_labels)
            
            # D. Backward Pass
            loss.backward()
            optimizer.step()
            
            # E. Metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += sensor_labels.size(0)
            correct_predictions += (predicted == sensor_labels).sum().item()
            
            # Print progress every 10 steps
            if (step + 1) % 10 == 0:
                print(f"   Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{step+1}/{min_len}] Loss: {loss.item():.4f}")

        # End of Epoch Stats
        epoch_acc = 100 * correct_predictions / total_samples
        avg_loss = running_loss / min_len
        duration = time.time() - start_time
        
        print(f"✨ Epoch {epoch+1} Summary: Loss={avg_loss:.4f} | Acc={epoch_acc:.2f}% | Time={duration:.1f}s")
        
        # 5. Save Checkpoint
        save_path = os.path.join(CHECKPOINT_DIR, f'impact_lung_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), save_path)
        print(f"   💾 Model saved to {save_path}")

    print("\n[4/4] Training Complete! 🎓")

if __name__ == "__main__":
    train_impact_lung()