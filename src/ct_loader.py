import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Path Configuration
CT_IMG_DIR = './data/ct_scans/images'
CT_MASK_DIR = './data/ct_scans/masks'
IMG_SIZE = 256

class CTSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # --- FIX: Added .tif and .tiff to allowed extensions ---
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        
        self.images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)])
        
        # Verify corresponding masks exist
        self.valid_images = []
        for img_name in self.images:
            if os.path.exists(os.path.join(mask_dir, img_name)):
                self.valid_images.append(img_name)
        
        print(f"✅ [CT Modality] Found {len(self.valid_images)} CT scans with matching masks.")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_name = self.valid_images[idx]
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Convert to Grayscale (L)
        try:
            image = Image.open(img_path).convert("L")
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            print(f"❌ Error loading {img_name}: {e}")
            # Return dummy black image on failure to prevent crash
            return torch.zeros(1, IMG_SIZE, IMG_SIZE), torch.zeros(1, IMG_SIZE, IMG_SIZE)
        
        if self.transform:
            # Resize
            resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
            image = resize(image)
            mask = resize(mask)
            
            # To Tensor
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
            mask = to_tensor(mask)
            
        return image, mask

def get_ct_loader():
    if not os.path.exists(CT_IMG_DIR):
        print(f"⚠️  Warning: CT folder not found at {CT_IMG_DIR}")
        return None
        
    dataset = CTSegmentationDataset(CT_IMG_DIR, CT_MASK_DIR, transform=True)
    
    # Safety check
    if len(dataset) == 0:
        return None
        
    return DataLoader(dataset, batch_size=4, shuffle=True)