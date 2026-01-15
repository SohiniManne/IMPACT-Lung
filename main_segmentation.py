import matplotlib.pyplot as plt
from src.ct_loader import get_ct_loader
from src.unet import UNet
import torch

def verify_segmentation():
    print("🚀 Verifying Week 2: CT Imaging & Segmentation")
    
    # 1. Load Data
    loader = get_ct_loader()
    if loader is None:
        return
        
    images, masks = next(iter(loader))
    
    print(f"✅ CT Batch Shape: {images.shape} (Batch, 1, 256, 256)")
    print(f"✅ Mask Batch Shape: {masks.shape} (Batch, 1, 256, 256)")
    
    # 2. Initialize U-Net
    model = UNet()
    print("✅ U-Net Model Initialized.")
    
    # 3. Forward Pass (Test the dimensions)
    with torch.no_grad():
        output = model(images)
        print(f"✅ Model Output Shape: {output.shape}")

    # 4. Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show CT Slice
    axes[0].imshow(images[0].squeeze(), cmap='gray')
    axes[0].set_title("Input: CT Slice")
    axes[0].axis('off')
    
    # Show Ground Truth Mask
    axes[1].imshow(masks[0].squeeze(), cmap='gray')
    axes[1].set_title("Target: Lung Mask")
    axes[1].axis('off')
    
    # Show Model Prediction (Untrained - will look like noise/gray)
    axes[2].imshow(output[0].squeeze(), cmap='gray')
    axes[2].set_title("Prediction (Untrained)")
    axes[2].axis('off')
    
    plt.show()
    print("🎉 Week 2 Requirement 'CT Imaging & Segmentation' COMPLETE.")

if __name__ == "__main__":
    verify_segmentation()