import torch
import time
import numpy as np
from src.attention_model import AttentionFusionModel

def benchmark_speed():
    print("🚀 Starting Week 7: Latency & Throughput Test")
    device = torch.device("cpu") # Benchmarking on CPU is standard for latency tests
    
    # Initialize Model
    model = AttentionFusionModel(num_classes=2).to(device)
    model.eval()
    
    # --- THE FIX IS HERE ---
    # We changed channels to 3 (for ResNet) AND emr_features to 4 (to match your Encoder)
    dummy_img = torch.randn(1, 3, 224, 224).to(device) 
    dummy_sen = torch.randn(1, 1, 187).to(device)
    dummy_emr = torch.randn(1, 4).to(device)  # <--- CHANGED FROM 12 TO 4
    
    print("   Warming up engine...")
    try:
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_img, dummy_sen, dummy_emr)
    except RuntimeError as e:
        print(f"❌ Warmup Failed: {e}")
        return
        
    print("   Running benchmark (50 iterations)...")
    iterations = 50
    times = []
    
    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            _ = model(dummy_img, dummy_sen, dummy_emr)
            end = time.time()
            times.append((end - start) * 1000) # Convert to ms
            
    avg_time = np.mean(times)
    fps = 1000 / avg_time
    
    print("\n⏱️  --- LATENCY REPORT ---")
    print(f"   Average Time per Patient: {avg_time:.2f} ms")
    print(f"   Throughput: {fps:.2f} patients/second")
    
    if avg_time < 100:
        print("✅ Status: REAL-TIME CAPABLE (<100ms)")
    else:
        print("⚠️ Status: BATCH PROCESSING ONLY")

if __name__ == "__main__":
    benchmark_speed()