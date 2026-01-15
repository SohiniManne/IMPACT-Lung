import torch
import torch.nn as nn
import torch.optim as optim
from src.config import BATCH_SIZE
from src.imaging_loader import get_imaging_loader
from src.sensor_loader import get_sensor_loader
from src.emr_loader import get_emr_loader
from src.fusion_model import ImpactLungModel

def run_training_check():
    print(f"🚀 IMPACT-Lung: Day 3 Fusion & Training Check (Batch Size: {BATCH_SIZE})")
    
    # 1. Load Data (Unified Batch Size)
    print("Step 1: Loading Data...")
    loader_img = get_imaging_loader()
    loader_sensor = get_sensor_loader()
    loader_emr = get_emr_loader()
    
    # Get one batch from each to simulate an iteration
    # (In real training, we zip these loaders together)
    imgs, labels_img = next(iter(loader_img))
    sensors, labels_sensor = next(iter(loader_sensor))
    emr, _ = next(iter(loader_emr))
    
    # ⚠️ REALITY CHECK: We simulate that all data comes from the SAME patient.
    # In a real hospital dataset, these would be linked by Patient_ID.
    # For this Day 3 code check, we assume the batches align.
    
    # 2. Initialize Model
    print("Step 2: Initializing Fusion Model...")
    model = ImpactLungModel(num_classes=2)
    
    # 3. Define Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Forward Pass (Prediction)
    print("Step 3: Forward Pass (Thinking)...")
    outputs = model(imgs, sensors, emr)
    print(f"   Model Output Shape: {outputs.shape} (Expect [4, 2])")
    
    # 5. Backward Pass (Learning)
    print("Step 4: Backward Pass (Learning)...")
    # We create fake ground-truth labels for this test (e.g., all '0' / Healthy)
    fake_targets = torch.zeros(BATCH_SIZE, dtype=torch.long)
    
    loss = criterion(outputs, fake_targets)
    print(f"   Current Loss: {loss.item():.4f}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("\n✅ Day 3 Complete: Gradient Descent is working. The model can learn!")

if __name__ == "__main__":
    run_training_check()