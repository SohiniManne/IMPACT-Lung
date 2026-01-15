from .config import EMR_FILE, BATCH_SIZE
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from .config import EMR_FILE, BATCH_SIZE

class ClinicalDataset(Dataset):
    def __init__(self, csv_file):
        try:
            # Load the raw clinical table
            self.df = pd.read_csv(csv_file)
            
            # Debug: Print found columns
            # print(f"🔍 DEBUG: Columns found: {list(self.df.columns)}")
            
            # 1. Normalize columns to uppercase
            self.df.columns = [c.upper().strip() for c in self.df.columns]
            
            # 2. Define target columns
            target_cols = ['ADMISSION_TYPE', 'INSURANCE', 'ETHNICITY', 'DIAGNOSIS']
            
            # 3. Handle missing columns
            missing_cols = [c for c in target_cols if c not in self.df.columns]
            if missing_cols:
                # print(f"⚠️  WARNING: Missing columns {missing_cols}. Filling with placeholders.")
                for c in missing_cols:
                    self.df[c] = "UNKNOWN"
            
            self.features = self.df[target_cols].copy()
            self.features = self.features.fillna("UNKNOWN")
            
            # 4. Encode Features
            self.label_encoders = {}
            for col in self.features.columns:
                le = LabelEncoder()
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
        
        # --- THE FIX IS HERE ---
        # Return (Data, Label). We use 0 as a dummy label for now.
        return torch.tensor(row, dtype=torch.float32), 0

def get_emr_loader():
    dataset = ClinicalDataset(EMR_FILE)
    if len(dataset) == 0:
        return None
    
    # FIX: Add drop_last=True to prevent batch_size=1 errors
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)