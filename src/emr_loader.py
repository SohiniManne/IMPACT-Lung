import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Configuration
EMR_FILE = './data/emr/ADMISSIONS.csv'
BATCH_SIZE = 4

class ClinicalDataset(Dataset):
    def __init__(self, csv_file):
        try:
            # Load the raw clinical table
            self.df = pd.read_csv(csv_file)
            
            # --- DEBUGGING: Print what we actually found ---
            print(f"\n🔍 DEBUG: Columns found in CSV: {list(self.df.columns)}")
            
            # 1. Normalize columns to uppercase to avoid case-sensitivity errors
            self.df.columns = [c.upper().strip() for c in self.df.columns]
            
            # 2. Define the target columns we WANT
            target_cols = ['ADMISSION_TYPE', 'INSURANCE', 'ETHNICITY', 'DIAGNOSIS']
            
            # 3. Check if they exist
            missing_cols = [c for c in target_cols if c not in self.df.columns]
            if missing_cols:
                print(f"⚠️  WARNING: Missing columns {missing_cols}. Using placeholders.")
                # Create fake columns for any missing ones to prevent crashing
                for c in missing_cols:
                    self.df[c] = "UNKNOWN"
            
            # --- FEATURE SELECTION ---
            self.features = self.df[target_cols].copy()
            
            # --- DATA CLEANING ---
            self.features = self.features.fillna("UNKNOWN")
            
            # --- ENCODING ---
            self.label_encoders = {}
            for col in self.features.columns:
                le = LabelEncoder()
                # Convert to string to handle mixed types
                self.features[col] = le.fit_transform(self.features[col].astype(str))
                self.label_encoders[col] = le
            
            self.data_matrix = self.features.values.astype(float)
            print(f"✅ [EMR Modality] Loaded. Processed {len(self.df)} clinical records.")
            
        except FileNotFoundError:
            print(f"❌ Error: Could not find EMR file at {csv_file}")
            self.data_matrix = []
        except Exception as e:
            print(f"❌ Critical Error in EMR Loader: {e}")
            self.data_matrix = []

    def __len__(self):
        return len(self.data_matrix)

    def __getitem__(self, idx):
        row = self.data_matrix[idx]
        return torch.tensor(row, dtype=torch.float32)

def get_emr_loader():
    dataset = ClinicalDataset(EMR_FILE)
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)