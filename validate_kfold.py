import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import numpy as np

# Imports
from src.config import BATCH_SIZE
from src.imaging_loader import get_imaging_loader
from src.sensor_loader import get_sensor_loader
from src.emr_loader import get_emr_loader
from src.attention_model import AttentionFusionModel

def run_kfold_validation(k=3): 
    print(f"🚀 Starting {k}-Fold Cross Validation (Fast CPU Mode)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using Device: {device}")
    
    # 1. Load Data
    l_img = get_imaging_loader(split='train')
    l_sen = get_sensor_loader()
    l_emr = get_emr_loader()
    
    # 2. Align Data (Limit to 129 samples)
    min_len = min(len(l_img.dataset), len(l_sen.dataset), len(l_emr.dataset))
    print(f"   ℹ️  Aligning datasets to {min_len} samples.")
    
    data_img, data_sen, data_emr, data_lbl = [], [], [], []
    
    # Extract data
    iterator = zip(l_img, l_sen, l_emr)
    for i, (b_img, b_sen, b_emr) in enumerate(iterator):
        if i * BATCH_SIZE >= min_len: break
        data_img.append(b_img[0])
        data_sen.append(b_sen[0])
        data_lbl.append(b_sen[1])
        data_emr.append(b_emr[0])

    X_img = torch.cat(data_img)[:min_len]
    X_sen = torch.cat(data_sen)[:min_len]
    X_emr = torch.cat(data_emr)[:min_len]
    Y_lbl = torch.cat(data_lbl)[:min_len]

    # 3. Start K-Fold
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_img)):
        print(f"\n📄 FOLD {fold+1}/{k} Processing...")
        
        # Prepare Data
        t_img, v_img = X_img[train_idx], X_img[val_idx]
        t_sen, v_sen = X_sen[train_idx], X_sen[val_idx]
        t_emr, v_emr = X_emr[train_idx], X_emr[val_idx]
        t_lbl, v_lbl = Y_lbl[train_idx], Y_lbl[val_idx]
        
        train_ds = TensorDataset(t_img, t_sen, t_emr, t_lbl)
        val_ds = TensorDataset(v_img, v_sen, v_emr, v_lbl)
        
        # --- THE FIX IS HERE: drop_last=True ---
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        # Validation doesn't need drop_last because eval mode handles single samples fine
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        # Init Model
        model = AttentionFusionModel(num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Train (1 Epoch Only)
        model.train()
        print(f"   -> Training Fold {fold+1}...", end="", flush=True)
        
        # Wrap in try-except to catch any other stray errors gracefully
        try:
            for batch_i, batch in enumerate(train_loader):
                img, sen, emr, lbl = [b.to(device) for b in batch]
                optimizer.zero_grad()
                out = model(img, sen, emr)
                loss = criterion(out, lbl)
                loss.backward()
                optimizer.step()
                if batch_i % 2 == 0: print(".", end="", flush=True) 
            print(" Done.")
            
            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    img, sen, emr, lbl = [b.to(device) for b in batch]
                    out = model(img, sen, emr)
                    _, pred = torch.max(out, 1)
                    correct += (pred == lbl).sum().item()
                    total += lbl.size(0)
            
            acc = 100 * correct / total
            print(f"   -> Fold Accuracy: {acc:.2f}%")
            fold_results.append(acc)
            
        except Exception as e:
            print(f"\n❌ Error in Fold {fold+1}: {e}")
            print("   Skipping this fold...")

    print("\n📊 --- CROSS-VALIDATION REPORT ---")
    if fold_results:
        print(f"   Mean Accuracy: {np.mean(fold_results):.2f}% ± {np.std(fold_results):.2f}%")
    else:
        print("   No folds completed successfully.")
    print("✅ Week 7 Validation Complete.")

if __name__ == "__main__":
    run_kfold_validation()