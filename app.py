import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import cv2
import shap
from PIL import Image

# Import your system
from src.config import IMG_SIZE
from src.imaging_loader import get_imaging_loader
from src.sensor_loader import get_sensor_loader
from src.emr_loader import get_emr_loader
from src.attention_model import AttentionFusionModel

# Page Configuration
st.set_page_config(page_title="IMPACT-Lung Dashboard", layout="wide", page_icon="🫁")

# --- HELPER CLASS: GRAD-CAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, sen, emr, class_idx=None):
        self.model.zero_grad()
        output, _ = self.model(x, sen, emr, return_weights=True)
        if class_idx is None: class_idx = torch.argmax(output, dim=1)
        score = output[0, class_idx]
        score.backward()
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2))
        heatmap = torch.zeros(activations.shape[1:], device=gradients.device)
        for i, w in enumerate(weights): heatmap += w * activations[i]
        heatmap = F.relu(heatmap)
        return heatmap.detach().cpu().numpy()

# --- HELPER CLASS: SHAP WRAPPER ---
class ModelWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def predict(self, data_combined):
        tensor_data = torch.tensor(data_combined, dtype=torch.float32).to(self.device)
        n_sensor = 187
        sens = tensor_data[:, :n_sensor].unsqueeze(1)
        emr = tensor_data[:, n_sensor:]
        dummy_img = torch.zeros(tensor_data.shape[0], 3, 224, 224).to(self.device)
        
        # FIX: Ensure model is in eval mode during SHAP too
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(dummy_img, sens, emr)
            probs = torch.softmax(outputs, dim=1)
        return probs[:, 1].cpu().numpy()

# --- HEADER ---
st.title("🫁 IMPACT-Lung: Multimodal AI System")
st.markdown("**System Status:** ✅ Online | **Week 8 Complete:** Explainable AI Integrated")
st.divider()

# --- SIDEBAR ---
st.sidebar.header("🕹️ Control Panel")
device = torch.device("cpu")

# WEEK 7 METRICS
st.sidebar.subheader("📊 System Benchmarks")
col_a, col_b = st.sidebar.columns(2)
col_a.metric("Accuracy", "100%", delta="Week 7")
col_b.metric("Latency", "24 ms", delta="Real-time")

# ABLATION
st.sidebar.divider()
st.sidebar.subheader("📉 Ablation Simulation")
use_img = st.sidebar.checkbox("Enable X-Ray", value=True)
use_sen = st.sidebar.checkbox("Enable Sensors", value=True)
use_emr = st.sidebar.checkbox("Enable EMR", value=True)

@st.cache_resource
def load_system():
    model = AttentionFusionModel(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load('./checkpoints/impact_lung_optimized.pth', map_location=device))
        model.eval() # <--- THIS WAS THE MISSING LINE FIXING THE CRASH
        status = "✅ Model Online"
    except:
        status = "❌ Model Error"
        model = None
    
    l_img = get_imaging_loader(split='test')
    l_sen = get_sensor_loader()
    l_emr = get_emr_loader()
    return model, l_img, l_sen, l_emr, status

model, loader_img, loader_sensor, loader_emr, status_msg = load_system()
st.sidebar.caption(status_msg)

if st.sidebar.button("🔄 Load New Patient"):
    st.cache_data.clear()

# --- MAIN LOGIC ---
if model:
    # Ensure model is in eval mode globally
    model.eval()

    # Get Data
    imgs, _ = next(iter(loader_img))
    sensors, labels = next(iter(loader_sensor))
    emr, _ = next(iter(loader_emr))
    
    # Pick Patient 0
    p_img = imgs[0].unsqueeze(0).to(device)
    p_sensor = sensors[0].unsqueeze(0).to(device)
    p_emr = emr[0].unsqueeze(0).to(device)
    
    # 1. DISPLAY DATA
    st.subheader("1. Patient Data Acquisition")
    c1, c2, c3 = st.columns(3)
    with c1:
        if use_img:
            disp_img = p_img.squeeze().permute(1, 2, 0).cpu().numpy()
            disp_img = (disp_img * 0.229) + 0.485
            disp_img = np.clip(disp_img, 0, 1)
            st.image(disp_img, caption="Chest X-Ray", use_container_width=True)
        else: st.warning("No Signal")
    with c2:
        if use_sen:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.plot(p_sensor.squeeze().cpu().numpy(), color='green')
            ax.axis('off')
            st.pyplot(fig)
            st.caption("Live ECG Rhythm")
        else: st.warning("No Signal")
    with c3:
        if use_emr:
            st.dataframe(pd.DataFrame(p_emr.squeeze().cpu().numpy(), columns=["Value"]).T)
        else: st.warning("No Record")

    st.divider()

    # 2. DIAGNOSIS
    st.subheader("2. AI Diagnosis")
    if st.button("🚀 Run Clinical Analysis"):
        # Double check eval mode prevents crash
        model.eval() 

        final_img = p_img if use_img else torch.zeros_like(p_img)
        final_sen = p_sensor if use_sen else torch.zeros_like(p_sensor)
        final_emr = p_emr if use_emr else torch.zeros_like(p_emr)

        start = time.time()
        with torch.no_grad():
            output, attn = model(final_img, final_sen, final_emr, return_weights=True)
        lat = (time.time() - start) * 1000

        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs).item()
        conf = probs[0][pred].item() * 100

        c1, c2 = st.columns([1, 2])
        with c1:
            if pred == 1: st.error(f"🛑 PATHOLOGY DETECTED\n\nConf: {conf:.2f}%")
            else: st.success(f"✅ PATIENT NORMAL\n\nConf: {conf:.2f}%")
            st.caption(f"⚡ Latency: {lat:.2f} ms")
        
        with c2:
            st.bar_chart(pd.DataFrame({'Importance': attn[0].mean(0).detach().cpu().numpy()}, index=['X-Ray', 'Sensor', 'EMR']))

        # --- EXTENSION 1: GRAD-CAM ---
        st.divider()
        st.subheader("3. Explainable AI (XAI) Analysis")
        
        with st.expander("🔍 Visual Explanation (Grad-CAM)"):
            if use_img:
                try:
                    target = model.img_encoder.resnet.layer4[1].conv2
                    cam = GradCAM(model, target)
                    heatmap = cam(final_img, final_sen, final_emr, class_idx=1)
                    
                    # Process for display
                    heatmap = cv2.resize(heatmap, (224, 224))
                    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                    
                    orig = (disp_img - disp_img.min()) / (disp_img.max() - disp_img.min())
                    overlay = 0.6 * orig + 0.4 * heatmap
                    
                    xc1, xc2 = st.columns(2)
                    with xc1: st.image(heatmap, caption="Attention Heatmap", clamp=True, use_container_width=True)
                    with xc2: st.image(overlay, caption="Overlay on Lungs", clamp=True, use_container_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM Failed: {e}")
            else:
                st.info("Enable X-Ray to see Grad-CAM.")

        # --- EXTENSION 2: SHAP ---
        with st.expander("📊 Data Explanation (SHAP)"):
            if use_sen or use_emr:
                with st.spinner("Calculating Feature Importance..."):
                    # Prepare Background
                    sen_batch_np = sensors.squeeze(1).numpy()
                    emr_batch_np = emr.numpy()
                    bg_data = np.hstack([sen_batch_np, emr_batch_np])
                    
                    # Create Explainer
                    wrapper = ModelWrapper(model, device)
                    k = min(len(bg_data), 5)
                    explainer = shap.KernelExplainer(wrapper.predict, shap.kmeans(bg_data, k))
                    
                    # Explain Current Patient
                    curr_sen = final_sen.squeeze(1).cpu().numpy()
                    curr_emr = final_emr.cpu().numpy()
                    curr_data = np.hstack([curr_sen, curr_emr])
                    
                    shap_values = explainer.shap_values(curr_data, nsamples=50)
                    
                    # Plot
                    feat_names = [f"ECG_{i}" for i in range(187)] + ["Age", "Gender", "Temp", "SpO2"]
                    fig_shap = plt.figure()
                    shap.summary_plot(shap_values, curr_data, feature_names=feat_names, show=False)
                    st.pyplot(fig_shap)
            else:
                st.info("Enable Sensors/EMR to see SHAP.")