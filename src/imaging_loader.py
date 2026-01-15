from .config import IMG_CSV, IMG_DIR, IMG_SIZE, BATCH_SIZE # Add BATCH_SIZE
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# Import BATCH_SIZE so we stay synced with the other modalities
from .config import IMG_CSV, IMG_DIR, IMG_SIZE, BATCH_SIZE

class PulmonaryXRayDataset(Dataset):
    def __init__(self, transform=None):
        self.root_dir = IMG_DIR
        self.transform = transform
        
        # Load the CSV metadata
        if not os.path.exists(IMG_CSV):
            raise FileNotFoundError(f"Missing Metadata CSV at: {IMG_CSV}")
            
        self.df = pd.read_csv(IMG_CSV)
        
        # REAL DATA FILTERING:
        # We only keep rows where the image file actually exists locally
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
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ Error loading image {img_name}: {e}")
            # Return a black image if file is corrupt (prevents crash)
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        # 3. Get Label (Ground Truth)
        label_str = self.valid_data.iloc[idx]['Finding Labels']
        
        # 4. Apply Transforms (IoMT Edge Processing)
        if self.transform:
            image = self.transform(image)
            
        # Return a dummy label index (0) for now, as we focus on the pipeline
        # Later we will parse the strings "Pneumonia|Infiltration" into vectors
        return image, 0 

def get_imaging_loader(split='train'):
    # --- WEEK 2 UPGRADE: Data Augmentation ---
    if split == 'train':
        # For training, we scramble images slightly to make the model "tougher"
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip L/R
            transforms.RandomRotation(degrees=10),   # Rotate slightly (-10 to 10 deg)
            transforms.ColorJitter(brightness=0.1, contrast=0.1), # Slight lighting change
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # For testing/validation, we keep it clean and simple
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    dataset = PulmonaryXRayDataset(transform=transform)
    
    if len(dataset) == 0:
        print("⚠️  WARNING: No images found. Check /data/imaging folder.")
        return None
        
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(split=='train'))
    return loader