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
from torchvision import transforms

# Import your system
from src.attention_model import AttentionFusionModel
from src.imaging_loader import get_imaging_loader
from src.sensor_loader import get_sensor_loader
from src.emr_loader import get_emr_loader

# Page Configuration
st.set_page_config(page_title="IMPACT-Lung Dashboard", layout="wide", page_icon="🫁")

# --- HELPER CLASSES ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]
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
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(dummy_img, sens, emr)
            probs = torch.softmax(outputs, dim=1)
        return probs[:, 1].cpu().numpy()

# --- UPLOAD HELPERS ---
def process_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0), img

def process_uploaded_csv(uploaded_file, expected_cols):
    df = pd.read_csv(uploaded_file, header=None)
    data = df.iloc[0, :].values.astype(np.float32)
    if len(data) < expected_cols: data = np.pad(data, (0, expected_cols - len(data)))
    else: data = data[:expected_cols]
    return torch.tensor(data).unsqueeze(0).unsqueeze(0)

# --- HEADER ---
st.title("🫁 IMPACT-Lung: Multimodal AI System")
st.markdown("**System Status:** ✅ Online |  Explainable AI Integrated")
st.divider()

# --- SIDEBAR (Preserving your Exact Look) ---
st.sidebar.header("🕹️ Control Panel")
device = torch.device("cpu")

# 1. MODE SELECTION (New Feature - Subtle placement)
mode = st.sidebar.radio("Operation Mode:", ["🎲 Random Test Case", "📤 Upload Patient Data"])
st.sidebar.divider()

# 2. SYSTEM BENCHMARKS (Preserved from your image)
st.sidebar.subheader("📊 System Benchmarks")
col_a, col_b = st.sidebar.columns(2)
col_a.metric("Val. Accuracy", "100%", delta="Top Tier")
col_b.metric("Avg Latency", "24 ms", delta="-0.01ms")
st.sidebar.caption("Validated via 3-Fold Cross-Validation on CPU.")
st.sidebar.divider()

# 3. ABLATION SIMULATION (Preserved)
st.sidebar.subheader("📉 Ablation Simulation")
use_img = st.sidebar.checkbox("Enable X-Ray", value=True)
use_sen = st.sidebar.checkbox("Enable Sensors", value=True)
use_emr = st.sidebar.checkbox("Enable EMR", value=True)

# LOAD MODEL
@st.cache_resource
def load_system():
    model = AttentionFusionModel(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load('./checkpoints/impact_lung_optimized.pth', map_location=device))
        model.eval()
        status = "✅ Model Online"
    except:
        status = "❌ Model Error"
        model = None
    
    # Load loaders only if needed for Random mode
    l_img = get_imaging_loader(split='test')
    l_sen = get_sensor_loader()
    l_emr = get_emr_loader()
    return model, l_img, l_sen, l_emr, status

model, loader_img, loader_sensor, loader_emr, status_msg = load_system()
st.sidebar.success(status_msg)

if mode == "🎲 Random Test Case":
    if st.sidebar.button("🔄 Load New Patient"):
        st.cache_data.clear()

# --- MAIN LOGIC ---
if model:
    # Initialize Defaults
    p_img, p_sen, p_emr = None, None, None
    disp_img_pil = None # For XAI display

    # ---------------------------------------------------------
    # CASE 1: RANDOM TEST MODE (Your Original Logic)
    # ---------------------------------------------------------
    if mode == "🎲 Random Test Case":
        # Get Data
        imgs, _ = next(iter(loader_img))
        sensors, labels = next(iter(loader_sensor))
        emr, _ = next(iter(loader_emr))
        
        # Pick Patient 0
        p_img = imgs[0].unsqueeze(0).to(device)
        p_sen = sensors[0].unsqueeze(0).to(device)
        p_emr = emr[0].unsqueeze(0).to(device)

        # Prepare Display Image
        disp_np = p_img.squeeze().permute(1, 2, 0).cpu().numpy()
        disp_np = (disp_np * 0.229) + 0.485
        disp_np = np.clip(disp_np, 0, 1)
        disp_img_pil = Image.fromarray((disp_np * 255).astype(np.uint8))

    # ---------------------------------------------------------
    # CASE 2: UPLOAD MODE (The New Feature)
    # ---------------------------------------------------------
    else:
        st.info("📂 Upload Patient Records")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            u_img = st.file_uploader("X-Ray Image", type=['jpg','png','jpeg'])
            if u_img:
                p_img, disp_img_pil = process_uploaded_image(u_img)
                p_img = p_img.to(device)
            else:
                p_img = torch.zeros(1, 3, 224, 224).to(device)
        
        with c2:
            u_sen = st.file_uploader("Sensor CSV", type=['csv'])
            if u_sen:
                p_sen = process_uploaded_csv(u_sen, 187).to(device)
            else:
                p_sen = torch.zeros(1, 1, 187).to(device)

        with c3:
            st.markdown("###### Clinical Vitals")
            age = st.number_input("Age", 0, 100, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            temp = st.number_input("Temp (°C)", 30.0, 45.0, 37.5)
            spo2 = st.number_input("SpO2 (%)", 50, 100, 98)
            
            g_val = 1.0 if gender == "Male" else 0.0
            p_emr = torch.tensor([age, g_val, temp, spo2], dtype=torch.float32).unsqueeze(0).to(device)

    # ---------------------------------------------------------
    # COMMON DISPLAY & DIAGNOSIS LOGIC
    # ---------------------------------------------------------
    st.subheader("1. Patient Data Acquisition")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if use_img and disp_img_pil:
            st.image(disp_img_pil, caption="Chest X-Ray", use_container_width=True)
        elif use_img and mode == "📤 Upload Patient Data":
            st.warning("Waiting for Upload...")
        else:
            st.warning("Signal Lost (Ablation)")

    with col2:
        if use_sen:
            if p_sen.abs().sum() > 0:
                fig, ax = plt.subplots(figsize=(4, 2))
                ax.plot(p_sen.squeeze().cpu().numpy(), color='green')
                ax.axis('off')
                st.pyplot(fig)
                st.caption("Live ECG Rhythm")
            elif mode == "📤 Upload Patient Data":
                 st.warning("Waiting for CSV...")
        else: st.warning("Signal Lost")

    with col3:
        if use_emr:
            st.dataframe(pd.DataFrame(p_emr.squeeze().cpu().numpy(), columns=["Value"]).T)
        else: st.warning("Record Unavailable")

    st.divider()

    # DIAGNOSIS
    st.subheader("2. Clinical Decision Support")
    if st.button("🚀 Run Diagnosis"):
        # Apply Ablation
        final_img = p_img if use_img else torch.zeros_like(p_img)
        final_sen = p_sen if use_sen else torch.zeros_like(p_sen)
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
            if pred == 1: st.error(f"🛑 **PATHOLOGY DETECTED**\n\nConfidence: {conf:.2f}%")
            else: st.success(f"✅ **PATIENT NORMAL**\n\nConfidence: {conf:.2f}%")
            st.caption(f"⚡ Inference Time: {lat:.2f} ms")
        
        with c2:
            st.bar_chart(pd.DataFrame({'Importance': attn[0].mean(0).detach().cpu().numpy()}, index=['X-Ray', 'Sensor', 'EMR']))

        # --- XAI EXTENSIONS ---
        st.divider()
        st.subheader("3. Explainable AI (XAI)")
        
        with st.expander("🔍 Visual Explanation (Grad-CAM)"):
            if use_img and disp_img_pil:
                try:
                    target = model.img_encoder.resnet.layer4[1].conv2
                    cam = GradCAM(model, target)
                    heatmap = cam(final_img, final_sen, final_emr, class_idx=1)
                    
                    # Resize & Colorize
                    heatmap = cv2.resize(heatmap, (224, 224))
                    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                    
                    # Overlay
                    orig_np = np.array(disp_img_pil.resize((224,224))) / 255.0
                    overlay = 0.6 * orig_np + 0.4 * heatmap
                    
                    xc1, xc2 = st.columns(2)
                    with xc1: st.image(heatmap, caption="Attention Heatmap", clamp=True, use_container_width=True)
                    with xc2: st.image(overlay, caption="Grad-CAM Overlay", clamp=True, use_container_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM Error: {e}")
            else:
                st.info("Load an image to see Heatmaps.")

        with st.expander("📊 Data Explanation (SHAP)"):
            if use_sen or use_emr:
                with st.spinner("Calculating Feature Importance..."):
                    # Create Background (Using random samples)
                    imgs, _ = next(iter(loader_img)) # Grab a batch
                    sensors, _ = next(iter(loader_sensor))
                    emr, _ = next(iter(loader_emr))
                    
                    bg_sen = sensors.squeeze(1).numpy()
                    bg_emr = emr.numpy()
                    bg_data = np.hstack([bg_sen, bg_emr])
                    
                    wrapper = ModelWrapper(model, device)
                    k = min(len(bg_data), 5)
                    explainer = shap.KernelExplainer(wrapper.predict, shap.kmeans(bg_data, k))
                    
                    # Explain Current
                    curr_sen = final_sen.squeeze(1).cpu().numpy()
                    curr_emr = final_emr.cpu().numpy()
                    curr_data = np.hstack([curr_sen, curr_emr])
                    
                    shap_values = explainer.shap_values(curr_data, nsamples=50)
                    feat_names = [f"ECG_{i}" for i in range(187)] + ["Age", "Gender", "Temp", "SpO2"]
                    
                    fig_shap = plt.figure()
                    shap.summary_plot(shap_values, curr_data, feature_names=feat_names, show=False)
                    st.pyplot(fig_shap)