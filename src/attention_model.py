import torch
import torch.nn as nn
from .encoders import ImageEncoder, SensorEncoder, EMREncoder

class AttentionFusionModel(nn.Module):
    def __init__(self, num_classes=2, embed_dim=64):
        super(AttentionFusionModel, self).__init__()
        
        # 1. Individual Encoders
        self.img_encoder = ImageEncoder()     # Out: 128
        self.sensor_encoder = SensorEncoder() # Out: 32
        self.emr_encoder = EMREncoder()       # Out: 16
        
        # 2. Projection Layers
        # Resize all features to 'embed_dim' (64)
        self.proj_img = nn.Linear(128, embed_dim)
        self.proj_sensor = nn.Linear(32, embed_dim)
        self.proj_emr = nn.Linear(16, embed_dim)
        
        # 3. Cross-Modal Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        
        # 4. Final Classifier
        # Input size = embed_dim * 3 (3 modalities)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, img, sensor, emr, return_weights=False):
        # A. Get Features
        f_img = self.img_encoder(img)          # [B, 128]
        f_sensor = self.sensor_encoder(sensor) # [B, 32]
        f_emr = self.emr_encoder(emr)          # [B, 16]
        
        # B. Project to common dimension
        p_img = self.proj_img(f_img)           # [B, 64]
        p_sensor = self.proj_sensor(f_sensor)  # [B, 64]
        p_emr = self.proj_emr(f_emr)           # [B, 64]
        
        # C. Stack features: [Batch, 3, 64]
        sequence = torch.stack([p_img, p_sensor, p_emr], dim=1)
        
        # D. Apply Attention
        # average_attn_weights=True gives us the attention map [Batch, Seq, Seq]
        attn_output, attn_weights = self.attention(sequence, sequence, sequence, average_attn_weights=True)
        
        # E. Flatten and Classify
        fused_vector = attn_output.reshape(attn_output.size(0), -1)
        output = self.classifier(fused_vector)
        
        if return_weights:
            return output, attn_weights
        
        return output