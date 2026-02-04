import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.imaging_loader import get_imaging_loader
from src.sensor_loader import get_sensor_loader
from src.emr_loader import get_emr_loader
from src.attention_model import AttentionFusionModel

def run_ablation():
    print("🚀 Starting Week 6: Ablation Study (Single vs. Multi-Modal)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Optimized Model
    model = AttentionFusionModel(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load('./checkpoints/impact_lung_optimized.pth', map_location=device))
    except:
        print("❌ Optimized model not found. Run train_weighted.py first.")
        return
        
    model.eval()
    
    # Load Data
    l_img = get_imaging_loader(split='test')
    l_sen = get_sensor_loader()
    l_emr = get_emr_loader()
    min_len = min(len(l_img), len(l_sen), len(l_emr))
    iterator = zip(l_img, l_sen, l_emr)
    
    # Store batches to reuse same data for fair comparison
    batches = []
    for step, batch in enumerate(iterator):
        if step >= min_len: break
        batches.append(batch)
        
    results = {}
    
    # --- EXPERIMENT 1: VISION ONLY ---
    print("   🧪 Testing Vision Only (Masking Sensors/EMR)...")
    correct = 0
    total = 0
    with torch.no_grad():
        for b_img, b_sen, b_emr in batches:
            img = b_img[0].to(device)
            # Create zeros with same shape as sensors/emr
            dummy_sensor = torch.zeros_like(b_sen[0]).to(device)
            dummy_emr = torch.zeros_like(b_emr[0]).to(device)
            labels = b_sen[1].to(device)
            
            out = model(img, dummy_sensor, dummy_emr)
            _, pred = torch.max(out.data, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    results['Vision Only'] = 100 * correct / total

    # --- EXPERIMENT 2: SENSOR ONLY ---
    print("   🧪 Testing Sensor Only (Masking Image/EMR)...")
    correct = 0
    total = 0
    with torch.no_grad():
        for b_img, b_sen, b_emr in batches:
            dummy_img = torch.zeros_like(b_img[0]).to(device)
            sensor = b_sen[0].to(device)
            dummy_emr = torch.zeros_like(b_emr[0]).to(device)
            labels = b_sen[1].to(device)
            
            out = model(dummy_img, sensor, dummy_emr)
            _, pred = torch.max(out.data, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    results['Sensor Only'] = 100 * correct / total

    # --- EXPERIMENT 3: FULL FUSION ---
    print("   🧪 Testing Full Fusion (All Modalities)...")
    correct = 0
    total = 0
    with torch.no_grad():
        for b_img, b_sen, b_emr in batches:
            img = b_img[0].to(device)
            sensor = b_sen[0].to(device)
            emr = b_emr[0].to(device)
            labels = b_sen[1].to(device)
            
            out = model(img, sensor, emr)
            _, pred = torch.max(out.data, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    results['IMPACT-Lung Fusion'] = 100 * correct / total

    # --- VISUALIZE RESULTS ---
    print("\n📊 --- ABLATION RESULTS ---")
    df = pd.DataFrame(list(results.items()), columns=['Modality', 'Accuracy'])
    print(df)
    
    plt.figure(figsize=(8, 5))
    colors = ['gray', 'gray', '#4A90E2'] # Highlight Fusion in Blue
    plt.bar(df['Modality'], df['Accuracy'], color=colors)
    plt.title("Impact of Multimodal Fusion vs Single Modality")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 110)
    for i, v in enumerate(df['Accuracy']):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.show()
    
    print("✅ Week 6 Requirements Complete.")

if __name__ == "__main__":
    run_ablation()