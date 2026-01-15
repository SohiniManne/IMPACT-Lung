from .config import SENSOR_FILE, BATCH_SIZE # Import from config
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
# IMPORT THE GLOBAL CONFIG
from .config import SENSOR_FILE, BATCH_SIZE 

class ECGDataset(Dataset):
    def __init__(self, csv_file):
        try:
            # We load only the first 1000 rows to keep it fast for testing
            self.data = pd.read_csv(csv_file, header=None, nrows=1000)
            print(f"✅ [Sensor Modality] Loaded. Found {len(self.data)} signals.")
        except FileNotFoundError:
            print(f"❌ Error: Could not find sensor file at {csv_file}")
            self.data = pd.DataFrame()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Row has 188 columns. Last column is label.
        row = self.data.iloc[idx].values.astype(np.float32)
        
        # Signal: First 187 points
        signal = row[:-1] 
        # Label: Last point (converted to integer)
        label = int(row[-1])
        
        # Reshape for PyTorch: [Channels, Time_Steps] -> [1, 187]
        signal_tensor = torch.tensor(signal).unsqueeze(0) 
        
        return signal_tensor, label

def get_sensor_loader():
    dataset = ECGDataset(SENSOR_FILE)
    if len(dataset) == 0:
        return None
    # USE THE GLOBAL BATCH_SIZE (4) HERE
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)