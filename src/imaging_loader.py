import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from .config import IMG_CSV, IMG_DIR, IMG_SIZE

class PulmonaryXRayDataset(Dataset):
    def __init__(self, transform=None):
        self.root_dir = IMG_DIR
        self.transform = transform
        
        # Load the CSV metadata
        if not os.path.exists(IMG_CSV):
            raise FileNotFoundError(f"Missing Metadata CSV at: {IMG_CSV}")
            
        self.df = pd.read_csv(IMG_CSV)
        
        # REAL DATA FILTERING:
        # We only keep rows where the image file actually exists in your folder.
        # This allows you to work with just a 1GB sample instead of the full 40GB dataset.
        self.df['exists'] = self.df['Image Index'].apply(
            lambda x: os.path.exists(os.path.join(self.root_dir, x))
        )
        self.valid_data = self.df[self.df['exists']].reset_index(drop=True)
        
        print(f"✅ [Imaging Modality] Loaded. Found {len(self.valid_data)} local images.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        # 1. Get Image Path
        img_name = self.valid_data.iloc[idx]['Image Index']
        img_path = os.path.join(self.root_dir, img_name)
        
        # 2. Load Image (IoMT Sensor Capture)
        image = Image.open(img_path).convert('RGB')
        
        # 3. Get Label (Ground Truth)
        label_str = self.valid_data.iloc[idx]['Finding Labels']
        
        # 4. Apply Transforms (IoMT Edge Processing)
        if self.transform:
            image = self.transform(image)
            
        return image, label_str

def get_imaging_loader():
    # Standard ResNet preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # Normalize to standard ImageNet means
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = PulmonaryXRayDataset(transform=transform)
    # If dataset is empty, warn the user
    if len(dataset) == 0:
        print("⚠️  WARNING: No images found. Did you unzip the NIH dataset into /data/imaging?")
        return None
        
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    return loader