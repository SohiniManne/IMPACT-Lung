import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from src.imaging_loader import get_imaging_loader

def verify_augmentation():
    print("🚀 Verifying Week 2: Data Augmentation & Preprocessing")
    
    # 1. Load the Training Data (where augmentation happens)
    # We pass split='train' to trigger the random flips/rotations
    loader = get_imaging_loader(split='train')
    
    if loader is None:
        print("❌ Error: Loader failed. Check your config path.")
        return

    # 2. Grab a single batch
    images, _ = next(iter(loader))
    
    print(f"✅ Loaded Batch Shape: {images.shape}")
    print("   (Batch_Size, Channels, Height, Width)")
    print("   Note: Channels=3 means we simulated RGB for the ResNet encoder.")

    # 3. Visualization Helper
    # We must 'un-normalize' the data to make it look like a normal X-ray again
    # (The model sees normalized numbers, but humans need 0-255 pixels)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    
    for i in range(len(images)):
        # Convert Tensor to Numpy: [C, H, W] -> [H, W, C]
        img = images[i].permute(1, 2, 0).numpy()
        
        # Un-normalize: input = (target * std) + mean
        img = std * img + mean
        img = np.clip(img, 0, 1) # Ensure colors stay valid
        
        # Display
        ax = axes[i] if len(images) > 1 else axes
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Sample {i+1}\n(Augmented)")

    plt.tight_layout()
    plt.show()
    print("✅ Verification Complete. If images look slightly rotated or flipped, Week 2 is SUCCESS.")

if __name__ == "__main__":
    verify_augmentation()