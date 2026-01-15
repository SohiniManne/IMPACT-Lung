import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import os

# Imports
from src.config import BATCH_SIZE
from src.imaging_loader import get_imaging_loader
from src.sensor_loader import get_sensor_loader
from src.emr_loader import get_emr_loader
from src.fusion_model import ImpactLungModel

def evaluate_model():
    print("🚀 Starting IMPACT-Lung Evaluation Protocol...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data (Test Mode)
    loader_img = get_imaging_loader(split='test')
    loader_sensor = get_sensor_loader()
    loader_emr = get_emr_loader()
    
    if loader_img is None or loader_sensor is None or loader_emr is None:
        print("❌ Error: One of the data loaders is empty.")
        return

    # Sync lengths
    min_len = min(len(loader_img), len(loader_sensor), len(loader_emr))
    iterator = zip(loader_img, loader_sensor, loader_emr)

    # 2. Load the Trained Model
    model = ImpactLungModel(num_classes=2).to(device)
    checkpoint_path = './checkpoints/impact_lung_epoch_5.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"✅ Loaded trained weights from: {checkpoint_path}")

    model.eval() # Set to evaluation mode
    
    all_preds = []
    all_labels = []

    print("   Running Inference...")
    
    with torch.no_grad():
        for step, (batch_img, batch_sensor, batch_emr) in enumerate(iterator):
            if step >= min_len: break
            
            # Prepare inputs
            imgs = batch_img[0].to(device)
            sensors = batch_sensor[0].to(device)
            labels = batch_sensor[1].to(device) # Ground Truth
            emr = batch_emr[0].to(device)
            
            # Prediction
            outputs = model(imgs, sensors, emr)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. Generate Metrics
    print("\n📊 --- CLASSIFICATION REPORT ---")
    
    # FIX: We explicitly pass labels=[0, 1] so it knows there are 2 classes
    # even if the batch only contains one of them.
    print(classification_report(all_labels, all_preds, labels=[0, 1], target_names=['Normal', 'Pathology'], zero_division=0))
    
    # 4. Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pathology'], 
                yticklabels=['Normal', 'Pathology'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('IMPACT-Lung Confusion Matrix')
    plt.show()
    
    print("✅ Evaluation Complete.")

if __name__ == "__main__":
    evaluate_model()