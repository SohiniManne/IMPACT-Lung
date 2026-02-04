import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
from src.attention_model import AttentionFusionModel
from src.emr_loader import get_emr_loader
from src.sensor_loader import get_sensor_loader

# --- CONFIGURATION ---
class ModelWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def predict(self, data_combined):
        """
        SHAP passes a numpy array 'data_combined' [Batch_Size, 191 features].
        We must split it back into Sensor (187) and EMR (4) to feed the model.
        """
        # 1. Convert to Tensor
        tensor_data = torch.tensor(data_combined, dtype=torch.float32).to(self.device)
        
        # 2. Split: [Batch, 191] -> Sensor [Batch, 1, 187], EMR [Batch, 4]
        n_sensor = 187
        sens = tensor_data[:, :n_sensor].unsqueeze(1) # Add channel dim: [B, 1, 187]
        emr = tensor_data[:, n_sensor:]               # [B, 4]
        
        # 3. Create Dummy Image
        batch_size = tensor_data.shape[0]
        dummy_img = torch.zeros(batch_size, 3, 224, 224).to(self.device)
        
        # 4. Run Model
        with torch.no_grad():
            outputs = self.model(dummy_img, sens, emr)
            probs = torch.softmax(outputs, dim=1)
            
        # Return ONLY Class 1 (Pathology) Probability
        return probs[:, 1].cpu().numpy()

def run_shap_analysis():
    print("🚀 Starting SHAP Analysis (Tabular Features)...")
    device = torch.device("cpu")
    
    # 1. Load Model
    model = AttentionFusionModel(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load('./checkpoints/impact_lung_optimized.pth', map_location=device))
        print("✅ Model Loaded.")
    except:
        print("⚠️ Checkpoint missing. Using random weights for demo.")
    model.eval()
    
    # 2. Prepare Data Background
    print("   Loading dataset samples...")
    l_sen = get_sensor_loader()
    l_emr = get_emr_loader()
    
    # Grab one batch
    sens_batch, _ = next(iter(l_sen)) 
    emr_batch, _ = next(iter(l_emr))   
    
    # Flatten Sensor data
    sens_flat = sens_batch.squeeze(1).numpy()
    emr_flat = emr_batch.numpy()
    
    # Combine columns
    background_data = np.hstack([sens_flat, emr_flat])
    
    # --- FIX FOR CRASH ---
    # Determine safe number of clusters (cannot be more than samples)
    n_samples = len(background_data)
    n_clusters = min(n_samples, 5) # Use at most 5, or fewer if batch is small
    print(f"   Creating background summary with k={n_clusters}...")
    
    # 3. Initialize SHAP Explainer
    wrapper = ModelWrapper(model, device)
    
    # Use robust kmeans
    background_summary = shap.kmeans(background_data, n_clusters)
    
    explainer = shap.KernelExplainer(wrapper.predict, background_summary)
    
    # 4. Explain a Test Patient
    print("   Calculating Feature Importance (this may take 10-20 seconds)...")
    test_instance = background_data[0:1] 
    
    shap_values = explainer.shap_values(test_instance, nsamples=100)
    
    # 5. Visualize
    print("   Generating Summary Plot...")
    
    feature_names = [f"ECG_{i}" for i in range(187)] + ["Age", "Gender", "Temp", "SpO2"]
    
    plt.figure()
    shap.summary_plot(
        shap_values,        
        test_instance, 
        feature_names=feature_names,
        show=False
    )
    plt.title("Why did AI predict Pathology? (Feature Impact)")
    plt.tight_layout()
    plt.show()
    
    print("✅ SHAP Analysis Complete. Check the popup window.")

if __name__ == "__main__":
    run_shap_analysis()