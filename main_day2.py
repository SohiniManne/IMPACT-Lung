import torch
from src.imaging_loader import get_imaging_loader
from src.sensor_loader import get_sensor_loader
from src.emr_loader import get_emr_loader
from src.encoders import ImageEncoder, SensorEncoder, EMREncoder

def run_pipeline_test():
    print("🚀 IMPACT-Lung: Day 2 Architecture Verification")
    print("-----------------------------------------------")
    
    # --- 1. Load Data (Day 1 Work) ---
    print("Step 1: Loading Real Data Batches...")
    img_loader = get_imaging_loader()
    sensor_loader = get_sensor_loader()
    emr_loader = get_emr_loader()
    
    # Get one batch from each
    images, _ = next(iter(img_loader))
    sensors, _ = next(iter(sensor_loader))
    emr_data = next(iter(emr_loader))
    
    print(f"   [Data] Images: {images.shape}")
    print(f"   [Data] Sensors: {sensors.shape}")
    print(f"   [Data] EMR:    {emr_data.shape}")

    # --- 2. Initialize Models (Day 2 Work) ---
    print("\nStep 2: Initializing Deep Learning Models...")
    # We define the output sizes (feature vector sizes)
    model_img = ImageEncoder(output_dim=128)
    model_sensor = SensorEncoder(output_dim=32)
    model_emr = EMREncoder(input_dim=emr_data.shape[1], output_dim=16)
    
    print("   ✅ Models initialized successfully.")

    # --- 3. Forward Pass (Processing) ---
    print("\nStep 3: Running Forward Pass (Feature Extraction)...")
    
    # Turn off gradient calculation for inference testing (saves memory)
    with torch.no_grad():
        # A. Process Image
        img_features = model_img(images)
        print(f"   [Output] Image Features:  {img_features.shape}  (Expect: [Batch, 128])")
        
        # B. Process Sensors
        sensor_features = model_sensor(sensors)
        print(f"   [Output] Sensor Features: {sensor_features.shape}   (Expect: [Batch, 32])")
        
        # C. Process EMR
        emr_features = model_emr(emr_data)
        print(f"   [Output] EMR Features:    {emr_features.shape}   (Expect: [Batch, 16])")

    print("\n🎉 Day 2 Complete: All Neural Networks are Operational.")

if __name__ == "__main__":
    run_pipeline_test()