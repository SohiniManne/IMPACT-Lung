from src.imaging_loader import get_imaging_loader
from src.sensor_loader import get_sensor_loader
from src.emr_loader import get_emr_loader
import matplotlib.pyplot as plt
import torchvision
import numpy as np

def visualize_complete_system(img_batch, img_labels, sensor_batch, sensor_labels, emr_batch):
    """
    The Dashboard: Shows X-ray, ECG, and Patient Data side-by-side.
    """
    plt.figure(figsize=(15, 5))
    
    # --- Panel 1: Imaging ---
    plt.subplot(1, 3, 1)
    inv_normalize = torchvision.transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img = inv_normalize(img_batch[0])
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.title(f"Modality A: Imaging\n{img_labels[0][:15]}...")
    plt.axis("off")
    
    # --- Panel 2: Sensors ---
    plt.subplot(1, 3, 2)
    signal = sensor_batch[0][0].numpy() 
    plt.plot(signal, color='green')
    plt.title(f"Modality B: Sensors (ECG)\nClass: {sensor_labels[0].item()}")
    plt.grid(True, alpha=0.3)
    
    # --- Panel 3: EMR Data ---
    plt.subplot(1, 3, 3)
    # Visualizing EMR is tricky because it's just numbers. 
    # We will plot a bar chart of the encoded feature values.
    features = emr_batch[0].numpy()
    feature_names = ['Adm. Type', 'Insurance', 'Ethnicity', 'Diagnosis']
    plt.bar(feature_names, features, color='purple')
    plt.title("Modality C: Clinical Records\n(Encoded Features)")
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🚀 IMPACT-Lung System Launch...")
    print("--------------------------------")
    
    # 1. Initialize Pipelines
    img_loader = get_imaging_loader()
    sensor_loader = get_sensor_loader()
    emr_loader = get_emr_loader()
    
    if img_loader and sensor_loader and emr_loader:
        # Fetch one batch from every modality
        images, i_labels = next(iter(img_loader))
        signals, s_labels = next(iter(sensor_loader))
        emr_data = next(iter(emr_loader))
        
        print(f"\n✅ SYSTEM STATUS: ONLINE")
        print(f"   [Image Input]   {images.shape}  (Visual)")
        print(f"   [Sensor Input]  {signals.shape} (Time-Series)")
        print(f"   [EMR Input]     {emr_data.shape}      (Structured Tabular)")
        
        print("\n🖥️  Generating IoMT Dashboard...")
        visualize_complete_system(images, i_labels, signals, s_labels, emr_data)
        
    else:
        print("❌ CRITICAL: One or more data pipelines failed.")