import torch
import torch.nn as nn
from .encoders import ImageEncoder, SensorEncoder, EMREncoder

class ImpactLungModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ImpactLungModel, self).__init__()
        
        # 1. Initialize the three specialized encoders
        self.img_encoder = ImageEncoder(output_dim=128)
        self.sensor_encoder = SensorEncoder(output_dim=32)
        self.emr_encoder = EMREncoder(output_dim=16)
        
        # 2. Fusion Layer
        # We combine the features: 128 + 32 + 16 = 176
        self.fusion_dim = 128 + 32 + 16
        
        # 3. Classifier Head (The Decision Maker)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes) 
            # Output: [Probability of Healthy, Probability of Pathology]
        )
        
    def forward(self, img_data, sensor_data, emr_data):
        # Step 1: Extract features independently
        v_img = self.img_encoder(img_data)       # [B, 128]
        v_sensor = self.sensor_encoder(sensor_data) # [B, 32]
        v_emr = self.emr_encoder(emr_data)       # [B, 16]
        
        # Step 2: Multimodal Fusion (Concatenation)
        # Combine them into one long vector
        fused_vector = torch.cat((v_img, v_sensor, v_emr), dim=1) # [B, 176]
        
        # Step 3: Final Prediction
        output = self.classifier(fused_vector) # [B, 2]
        
        return output