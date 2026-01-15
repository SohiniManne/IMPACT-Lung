import torch
import torch.nn as nn
from torchvision import models

# --- 1. IMAGING ENCODER (The "Eye") ---
class ImageEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(ImageEncoder, self).__init__()
        # ResNet18: Lightweight, standard for medical imaging
        self.resnet = models.resnet18(weights='DEFAULT')
        
        # Replace the final classification layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, output_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Input: [Batch, 3, 224, 224]
        x = self.resnet(x)
        x = self.relu(x)
        return x  # Output: [Batch, 128]

# --- 2. SENSOR ENCODER (The "Pulse") ---
class SensorEncoder(nn.Module):
    def __init__(self, input_dim=187, output_dim=32):
        super(SensorEncoder, self).__init__()
        
        # 1D Convolution to extract heartbeat spikes
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # CALCULATE FLATTENED SIZE:
        # Input 187 -> Pool(2) -> 93 time steps
        # Channels: 16
        # Flat size = 16 * 93 = 1488
        self.flatten_size = 16 * (input_dim // 2)
        
        # Final projection layer
        self.fc = nn.Linear(self.flatten_size, output_dim)

    def forward(self, x):
        # Input: [Batch, 1, 187]
        x = self.conv1(x)       # -> [Batch, 16, 187]
        x = self.relu(x)
        x = self.pool(x)        # -> [Batch, 16, 93]
        
        # Flatten: Turn [Batch, 16, 93] into [Batch, 1488]
        x = x.flatten(start_dim=1) 
        
        # Project to feature vector
        x = self.fc(x)          # -> [Batch, 32]
        return x

# --- 3. EMR ENCODER (The "Record") ---
class EMREncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=16):
        super(EMREncoder, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(32, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Input: [Batch, 4]
        return self.net(x) # Output: [Batch, 16]

if __name__ == "__main__":
    print("Testing Fixed Model Shapes...")
    sensor_model = SensorEncoder()
    fake_sensor = torch.randn(2, 1, 187) 
    print(f"Sensor Output: {sensor_model(fake_sensor).shape}")