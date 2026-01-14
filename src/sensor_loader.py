import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Configuration
SENSOR_FILE = './data/sensors/mitbih_test.csv' 
BATCH_SIZE = 16

class ECGDataset(Dataset):
    def __init__(self, csv_file):
        """
        MIT-BIH dataset is already normalized/segmented in this CSV version.
        Each row is one heartbeat signal (187 time steps).
        The last column is the label (0=Normal, 1=Arrhythmia, etc.)
        """
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
        # 1D-CNNs expect channels first.
        signal_tensor = torch.tensor(signal).unsqueeze(0) 
        
        return signal_tensor, label

def get_sensor_loader():
    dataset = ECGDataset(SENSOR_FILE)
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)