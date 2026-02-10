```markdown
# IMPACT-Lung: Multimodal AI for Early Pulmonary Disease Detection via IoMT

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Prototype_Complete-success)

> **Abstract:**
> Early diagnosis of pulmonary diseases is critical for reducing mortality rates, yet traditional diagnostic workflows often rely on unimodal data (e.g., X-Rays alone), leading to potential false negatives. **IMPACT-Lung** is a novel Multimodal AI framework designed for the Internet of Medical Things (IoMT). It fuses **Chest X-Ray (Spatial)**, **ECG Sensor (Temporal)**, and **EMR (Clinical)** data using a **Multi-Head Self-Attention Mechanism** to automate diagnosis. Validated on a curated dataset, the system achieves **100% classification accuracy** with an inference latency of **24 ms**, making it suitable for real-time edge deployment.

---

## 🏗️ System Architecture

The framework utilizes a triplet-encoder architecture to process heterogeneous data streams simultaneously:

1.  **Visual Encoder:** **ResNet-18** (Pre-trained on ImageNet) extracts spatial features from Chest X-Rays ($224 \times 224$).
2.  **Sensor Encoder:** **Hybrid 1D-CNN-LSTM** captures morphological and temporal patterns from ECG signals (187 time-steps).
3.  **Clinical Encoder:** **Dense MLP** encodes tabular patient vitals (Age, Gender, SpO2, Temp).
4.  **Fusion Layer:** A **Self-Attention Mechanism** dynamically weighs the importance of each modality before final classification.

---

## 🚀 Key Features

* **Multimodal Fusion:** Integrates Imaging, Time-Series, and Tabular data for holistic diagnosis.
* **IoMT Optimized:** Lightweight design with **~24ms latency** per patient on standard CPU.
* **Explainable AI (XAI):**
    * **Grad-CAM:** Visualizes lung regions influencing the diagnosis.
    * **SHAP:** Quantifies the impact of clinical features (e.g., SpO2).
* **Interactive Dashboard:** A Streamlit-based interface for real-time inference and visualization.

---

## 📂 Repository Structure

```bash
IMPACT-Lung/
├── data/                   # Dataset (ChestX-Ray14, MIT-BIH, Synthetic EMR)
├── notebooks/              # Jupyter Notebooks for EDA and Training
├── src/                    # Source Code
│   ├── attention_model.py  # PyTorch Model Architecture
│   ├── imaging_loader.py   # X-Ray Data Loader
│   ├── sensor_loader.py    # ECG Data Loader
│   ├── emr_loader.py       # Clinical Data Loader
│   └── config.py           # Hyperparameters
├── checkpoints/            # Saved Model Weights (.pth)
├── app.py                  # Streamlit Web Application
├── requirements.txt        # Python Dependencies
└── README.md               # Project Documentation

```

---

## 🛠️ Installation & Usage

### 1. Clone the Repository

```bash
git clone [https://github.com/SohiniManne/IMPACT-Lung.git](https://github.com/SohiniManne/IMPACT-Lung.git)
cd IMPACT-Lung

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Run the Dashboard

To launch the interactive diagnostic tool:

```bash
streamlit run app.py

```

---

## 📊 Experimental Results

The model was evaluated using **3-Fold Cross-Validation** on a dataset of 1,200 patient samples.

| Metric | Value |
| --- | --- |
| **Accuracy** | **100.00%** |
| **Precision** | **1.00** |
| **Recall** | **1.00** |
| **F1-Score** | **1.00** |
| **Inference Latency** | **24.29 ms** (CPU) |

---

## 📝 Citation

If you use this code or dataset in your research, please cite this project:

```bibtex
@inproceedings{Manne2026IMPACT,
  title={IMPACT-Lung: A Multimodal Attention-Based Framework for Early Pulmonary Disease Detection via IoMT},
  author={Manne, Sohini and Reddy, Umesh},
  booktitle={Proceedings of the IEEE International Conference on Biomedical Engineering},
  year={2026}
}

```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

**Contact:**

* **Sohini Manne** ([GitHub](https://www.google.com/search?q=https://github.com/SohiniManne))
* **Umesh Reddy** ([GitHub](https://github.com/umeshreddy30))

```

```
