import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Import your system
from src.config import IMG_SIZE
from src.imaging_loader import get_imaging_loader
from src.sensor_loader import get_sensor_loader
from src.emr_loader import get_emr_loader
from src.attention_model import AttentionFusionModel

# Page Configuration
st.set_page_config(page_title="IMPACT-Lung Dashboard", layout="wide")

# --- HEADER ---
st.title("🫁 IMPACT-Lung: Multimodal AI System")
st.markdown("""
**Current Status:** Week 6 Complete (Clinical Optimization & Ablation).  
**Model:** Weighted Attention Fusion (Minimizes False Negatives).
""")
st.divider()

# --- SIDEBAR (CONTROLS) ---
st.sidebar.header("🕹️ Simulation Panel")
device = torch.device("cpu")

# ABLATION CONTROLS (NEW FEATURE)
st.sidebar.subheader("📉 Ablation Test (Live)")
use_img = st.sidebar.checkbox("Enable X-Ray Data", value=True)
use_sen = st.sidebar.checkbox("Enable Sensor Data", value=True)
use_emr = st.sidebar.checkbox("Enable EMR Data", value=True)

@st.cache_resource
def load_system():
    # 1. Load Model (UPDATED TO WEEK 6 OPTIMIZED MODEL)
    model = AttentionFusionModel(num_classes=2).to(device)
    path = './checkpoints/impact_lung_optimized.pth'
    
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        status = "✅ Week 6 Model Online"
    except:
        status = "❌ Model Offline (Train Week 6 first)"
        model = None
        
    # 2. Load Data Loaders (Test Set)
    l_img = get_imaging_loader(split='test')
    l_sen = get_sensor_loader()
    l_emr = get_emr_loader()
    
    return model, l_img, l_sen, l_emr, status

model, loader_img, loader_sensor, loader_emr, status_msg = load_system()
st.sidebar.write(status_msg)

if st.sidebar.button("🔄 Load New Patient Case"):
    st.cache_data.clear()

# --- MAIN INFERENCE LOGIC ---
if model:
    # Get a random batch
    imgs, _ = next(iter(loader_img))
    sensors, labels = next(iter(loader_sensor))
    emr, _ = next(iter(loader_emr))
    
    # Pick Patient 0
    p_img = imgs[0].unsqueeze(0).to(device)
    p_sensor = sensors[0].unsqueeze(0).to(device)
    p_emr = emr[0].unsqueeze(0).to(device)
    true_label = labels[0].item()
    
    # --- DISPLAY PATIENT DATA ---
    st.subheader("1. Patient Data Acquisition")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Modality A: Imaging**")
        if use_img:
            disp_img = p_img.squeeze().permute(1, 2, 0).cpu().numpy()
            disp_img = (disp_img * 0.229) + 0.485 
            disp_img = np.clip(disp_img, 0, 1)
            st.image(disp_img, caption="Chest X-Ray", use_container_width=True)
        else:
            st.warning("⚠️ Signal Lost (Simulated)")

    with col2:
        st.info("**Modality B: IoMT Sensor**")
        if use_sen:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(p_sensor.squeeze().cpu().numpy(), color='green')
            ax.set_title("ECG Rhythm")
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("⚠️ Signal Lost (Simulated)")

    with col3:
        st.info("**Modality C: EMR Record**")
        if use_emr:
            st.dataframe(pd.DataFrame(p_emr.squeeze().cpu().numpy()).T)
        else:
            st.warning("⚠️ Record Unavailable")

    st.divider()

    # --- RUN AI PREDICTION ---
    st.subheader("2. AI Diagnosis & Explanation")
    
    if st.button("🚀 Run Clinical Analysis"):
        with st.spinner("Analyzing..."):
            # ABLATION LOGIC: Mask inputs with Zeros if unchecked
            final_img = p_img if use_img else torch.zeros_like(p_img)
            final_sen = p_sensor if use_sen else torch.zeros_like(p_sensor)
            final_emr = p_emr if use_emr else torch.zeros_like(p_emr)
            
            # Run Model
            output, attn_weights = model(final_img, final_sen, final_emr, return_weights=True)
            
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item() * 100
            attn_avg = attn_weights[0].mean(dim=0).detach().cpu().numpy()
            
        r_col1, r_col2 = st.columns([1, 2])
        
        with r_col1:
            st.markdown("### Diagnosis Result")
            if pred_class == 1:
                st.error(f"🛑 **PATHOLOGY DETECTED**\n\nConfidence: {confidence:.2f}%")
            else:
                st.success(f"✅ **PATIENT NORMAL**\n\nConfidence: {confidence:.2f}%")
            
            st.metric("Ground Truth", "Pathology" if true_label == 1 else "Normal")

        with r_col2:
            st.markdown("### 🧠 Live Attention Weights")
            att_df = pd.DataFrame({
                'Source': ['X-Ray', 'Sensor', 'EMR'],
                'Importance': [attn_avg[0], attn_avg[1], attn_avg[2]]
            })
            st.bar_chart(att_df.set_index('Source'))
            
            if not use_img and not use_sen and not use_emr:
                st.error("⚠️ All sensors disconnected. AI is guessing blindly.")

else:
    st.warning("Model not found.")