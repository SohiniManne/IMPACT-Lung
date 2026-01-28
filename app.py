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
**Objective:** Early detection of lung pathology using Data Fusion.  
**Modalities:** X-Ray Imaging (Vision) + IoMT Sensors (Time-Series) + EMR (Clinical Data).
""")
st.divider()

# --- SIDEBAR (CONTROLS) ---
st.sidebar.header("🕹️ Control Panel")
device = torch.device("cpu") # Use CPU for inference stability

@st.cache_resource
def load_system():
    # 1. Load Model
    model = AttentionFusionModel(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load('./checkpoints/impact_lung_attention.pth', map_location=device))
        model.eval()
        status = "✅ System Online"
    except:
        status = "❌ Model Offline (Train first)"
        model = None
        
    # 2. Load Data Loaders (Test Set)
    l_img = get_imaging_loader(split='test')
    l_sen = get_sensor_loader()
    l_emr = get_emr_loader()
    
    return model, l_img, l_sen, l_emr, status

model, loader_img, loader_sensor, loader_emr, status_msg = load_system()
st.sidebar.write(status_msg)

if st.sidebar.button("🔄 Load New Patient Case"):
    # Clear cache to fetch new random batch
    st.cache_data.clear()

# --- MAIN INFERENCE LOGIC ---
if model:
    # Get a random batch
    # We use iter() to grab the first batch of the shuffled loader
    imgs, _ = next(iter(loader_img))
    sensors, labels = next(iter(loader_sensor))
    emr, _ = next(iter(loader_emr))
    
    # Pick the first patient in the batch (Index 0)
    # (In a real app, we would select by Patient ID)
    p_img = imgs[0].unsqueeze(0).to(device)
    p_sensor = sensors[0].unsqueeze(0).to(device)
    p_emr = emr[0].unsqueeze(0).to(device)
    true_label = labels[0].item()
    
    # --- DISPLAY PATIENT DATA ---
    st.subheader("1. Patient Data Acquisition")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Modality A: Imaging (X-Ray)**")
        # Convert Tensor back to Image for display
        disp_img = p_img.squeeze().permute(1, 2, 0).cpu().numpy()
        disp_img = (disp_img * 0.229) + 0.485 # Un-normalize roughly
        disp_img = np.clip(disp_img, 0, 1)
        st.image(disp_img, caption="Preprocessed Input (224x224)", use_container_width=True)

    with col2:
        st.info("**Modality B: IoMT Sensor (ECG)**")
        # Plot the time-series signal
        fig, ax = plt.subplots(figsize=(4, 3))
        signal_data = p_sensor.squeeze().cpu().numpy()
        ax.plot(signal_data, color='green', linewidth=1.5)
        ax.set_title("Live Heart Rhythm Stream")
        ax.axis('off')
        st.pyplot(fig)
        st.caption(f"Sensor Label: {'Arrythmia' if true_label==1 else 'Normal'}")

    with col3:
        st.info("**Modality C: Clinical Record (EMR)**")
        # Show the raw feature vector
        emr_feats = p_emr.squeeze().cpu().numpy()
        df = pd.DataFrame(emr_feats.reshape(1, -1), columns=[f"F{i}" for i in range(len(emr_feats))])
        st.dataframe(df.style.highlight_max(axis=1), height=150)
        st.caption("Encoded Clinical Features (Normalized)")

    st.divider()

    # --- RUN AI PREDICTION ---
    st.subheader("2. AI Diagnosis & Explanation")
    
    if st.button("🚀 Run IMPACT-Lung Analysis"):
        with st.spinner("Fusing modalities..."):
            # Run Model with return_weights=True
            output, attn_weights = model(p_img, p_sensor, p_emr, return_weights=True)
            
            # Get Prediction
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item() * 100
            
            # Get Attention (Average across heads)
            # Shape is [Batch, Seq_Len, Seq_Len]. We want row 0 (Impact of each mod on Output)
            # Actually, standard attention maps are [Target_Seq, Source_Seq].
            # We just want to know: "How much did the model look at Img vs Sensor vs EMR?"
            # We take the diagonal or the average weight assigned to each input.
            # Simplified: Sum of weights for column 0, 1, 2
            attn_avg = attn_weights[0].mean(dim=0).detach().cpu().numpy() # [3] vector
            
        # --- RESULTS COLUMN ---
        r_col1, r_col2 = st.columns([1, 2])
        
        with r_col1:
            st.markdown("### Diagnosis Result")
            if pred_class == 1:
                st.error(f"🛑 **PATHOLOGY DETECTED**\n\nConfidence: {confidence:.2f}%")
            else:
                st.success(f"✅ **PATIENT NORMAL**\n\nConfidence: {confidence:.2f}%")
            
            st.metric("Ground Truth", "Pathology" if true_label == 1 else "Normal")

        # --- EXPLAINABILITY COLUMN ---
        with r_col2:
            st.markdown("### 🧠 Dynamic Attention Weights")
            st.write("How much did the AI rely on each data source?")
            
            # Bar Chart for Attention
            att_df = pd.DataFrame({
                'Source': ['X-Ray (Vision)', 'Sensor (Time-Series)', 'EMR (Clinical)'],
                'Importance': [attn_avg[0], attn_avg[1], attn_avg[2]]
            })
            
            st.bar_chart(att_df.set_index('Source'), color="#4A90E2")
            
            # Interpret
            winner = att_df.loc[att_df['Importance'].idxmax()]
            st.caption(f"💡 Insight: The model relied most heavily on **{winner['Source']}** for this specific case.")

else:
    st.warning("Please train the model to enable the dashboard.")