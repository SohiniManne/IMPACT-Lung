import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.attention_model import AttentionFusionModel
from src.imaging_loader import get_imaging_loader

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
        # Hook into the layer to catch gradients
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, sen, emr, class_idx=None):
        # 1. Forward Pass
        self.model.zero_grad()
        output, _ = self.model(x, sen, emr, return_weights=True)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        # 2. Backward Pass (Targeting the specific class)
        score = output[0, class_idx]
        score.backward()
        
        # 3. Generate Heatmap
        gradients = self.gradients[0]            # [C, H, W]
        activations = self.activations[0]        # [C, H, W]
        
        # Global Average Pooling of Gradients
        weights = torch.mean(gradients, dim=(1, 2)) # [C]
        
        # Weighted combination of activations
        heatmap = torch.zeros(activations.shape[1:], device=gradients.device)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        # ReLU and Normalization
        heatmap = F.relu(heatmap)
        heatmap = heatmap.detach().cpu().numpy()
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) + 1e-8)
        
        return heatmap

def run_xai_demo():
    print("🚀 Starting Week 8 Extension: Explainable AI (Grad-CAM)")
    device = "cpu"
    
    # Load Model
    model = AttentionFusionModel(num_classes=2).to(device)
    model.load_state_dict(torch.load('./checkpoints/impact_lung_optimized.pth', map_location=device))
    model.eval()
    
    # Target the last Convolutional Layer of ResNet
    # Structure: model -> img_encoder -> resnet -> layer4 -> [1] -> conv2
    target_layer = model.img_encoder.resnet.layer4[1].conv2
    grad_cam = GradCAM(model, target_layer)
    
    # Get a sample patient
    print("   Loading a test patient...")
    loader = get_imaging_loader(split='test')
    img_tensor, _ = next(iter(loader))
    img_tensor = img_tensor[0].unsqueeze(0).to(device) # [1, 3, 224, 224]
    
    # Dummy sensors (we focus on image explanation here)
    sen = torch.randn(1, 1, 187).to(device)
    emr = torch.randn(1, 4).to(device)
    
    # Generate Heatmap
    print("   Computing Gradients...")
    heatmap = grad_cam(img_tensor, sen, emr, class_idx=1) # Target 'Pathology' class
    
    # Visualization
    print("   Generating Visualization...")
    original_img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay
    superimposed = 0.6 * original_img + 0.4 * heatmap_colored
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original X-Ray")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap='jet')
    plt.title("Attention Heatmap")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed)
    plt.title("Grad-CAM Overlay")
    plt.axis('off')
    
    plt.show()
    print("✅ X-Ray Explanation Generated.")

if __name__ == "__main__":
    run_xai_demo()