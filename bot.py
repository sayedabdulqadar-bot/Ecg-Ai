"""
ECG Model - CNN Architecture for ECG Image Classification
Classifies ECG images into Normal vs Abnormal
"""

import torch
import torch.nn as nn


class ECGModel(nn.Module):
    """
    Convolutional Neural Network (CNN) for ECG image classification.
    Architecture:
    - 2 Convolutional layers with MaxPooling
    - 2 Fully connected layers
    - Dropout for regularization
    Output: 2 classes (Normal, Abnormal)
    """
    def __init__(self):
        super(ECGModel, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  # Adjust based on input size (224x224 -> 56x56 after 2 pools)
        self.fc2 = nn.Linear(512, 2)  # Output layer (2 classes: Normal, Abnormal)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output tensor of shape (batch_size, 2) with class logits
        """
        
        # Handle extra dimension if present: [B, C, H, W, 1] -> [B, C, H, W]
        if x.dim() == 5:
            x = x.squeeze(-1)
        
        # First conv block: Conv -> ReLU -> Pool
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        
        # Second conv block: Conv -> ReLU -> Pool
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        
        # Flatten the output from convolutional layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, features)
        
        # Fully connected layers
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)  # Apply dropout for regularization
        x = self.fc2(x)
        
        return x


if __name__ == "__main__":
    """Test the model architecture."""
    print("="*60)
    print("Testing ECGModel Architecture")
    print("="*60)
    
    # Create model instance
    model = ECGModel()
    
    # Test with dummy input (typical ECG image size)
    batch_size = 4
    channels = 1
    height = 224
    width = 224
    
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Model: {model.__class__.__name__}")
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output classes: 2 (Normal, Abnormal)")
    
    # Test softmax probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)
    print(f"\nSample probabilities (first batch item):")
    print(f"  Normal: {probabilities[0][0].item():.4f}")
    print(f"  Abnormal: {probabilities[0][1].item():.4f}")
    
    print("\n" + "="*60)
    print("✓ Model test passed!")
    print("="*60)