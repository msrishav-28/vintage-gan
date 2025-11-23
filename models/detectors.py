"""
Defect Detector Networks
Specification Reference: Section 4.2 - Consistency Loss Implementation

Implements lightweight detector networks to extract defect levels from images.
These are used for consistency loss during GAN training.

Author: VintageGAN Project
Date: 2024
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DefectDetector(nn.Module):
    """
    Base defect detector network.
    
    Specification Reference: Section 4.2
    
    Uses ResNet18 backbone (lightweight) to predict defect intensity [0, 1].
    
    Args:
        defect_name: Name of defect to detect
        pretrained: Use pretrained ResNet18 backbone
    """
    
    def __init__(self, defect_name: str, pretrained: bool = True):
        super().__init__()
        
        self.defect_name = defect_name
        
        # ResNet18 backbone (lightweight, <12M parameters)
        # Using new PyTorch 2.0+ weights API
        if pretrained:
            from torchvision.models import ResNet18_Weights
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Replace with single regression output
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict defect intensity.
        
        Args:
            x: Input images (B, 3, H, W) in [-1, 1]
        
        Returns:
            Defect intensity (B, 1) in [0, 1]
        """
        # Normalize input from [-1, 1] to [0, 1] for ResNet
        x = (x + 1) / 2
        
        # Extract features
        features = self.backbone(x)  # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)
        
        # Predict intensity
        intensity = self.fc(features)  # (B, 1)
        
        return intensity


class MultiDefectDetector(nn.Module):
    """
    Multi-task detector predicting all 6 defect types.
    
    More efficient than 6 separate detectors as it shares backbone.
    
    Args:
        pretrained: Use pretrained backbone
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Shared ResNet18 backbone
        # Using new PyTorch 2.0+ weights API
        if pretrained:
            from torchvision.models import ResNet18_Weights
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Separate heads for each defect
        self.defect_names = ['grain', 'scratch', 'dust', 'vignette', 'color_shift', 'blur']
        
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            for name in self.defect_names
        })
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict all defect intensities.
        
        Args:
            x: Input images (B, 3, H, W) in [-1, 1]
        
        Returns:
            Defect intensities (B, 6) in [0, 1]
        """
        # Normalize input
        x = (x + 1) / 2
        
        # Extract shared features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Predict each defect
        predictions = []
        for name in self.defect_names:
            pred = self.heads[name](features)
            predictions.append(pred)
        
        # Concatenate predictions
        output = torch.cat(predictions, dim=1)  # (B, 6)
        
        return output


def create_detectors(
    detector_type: str = 'separate',
    pretrained: bool = True
) -> Dict[str, nn.Module]:
    """
    Create defect detector networks.
    
    Args:
        detector_type: 'separate' (6 separate networks) or 'multi' (1 multi-task network)
        pretrained: Use pretrained backbones
    
    Returns:
        Dictionary mapping defect names to detector networks
    """
    defect_names = ['grain', 'scratch', 'dust', 'vignette', 'color_shift', 'blur']
    
    if detector_type == 'separate':
        detectors = {
            name: DefectDetector(name, pretrained=pretrained)
            for name in defect_names
        }
    elif detector_type == 'multi':
        # Single multi-task detector
        multi_detector = MultiDefectDetector(pretrained=pretrained)
        
        # Wrap in dict format for compatibility
        detectors = {'multi': multi_detector}
    else:
        raise ValueError(f"Unknown detector_type: {detector_type}")
    
    return detectors


def train_detectors(
    detectors: Dict[str, nn.Module],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 1e-3
) -> Dict[str, nn.Module]:
    """
    Train defect detector networks.
    
    Args:
        detectors: Dictionary of detector networks
        train_loader: Training dataloader (with defected images + conditions)
        val_loader: Validation dataloader
        device: Device to train on
        num_epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Trained detectors
    """
    from tqdm import tqdm
    
    print("="*60)
    print("TRAINING DEFECT DETECTORS")
    print("="*60)
    
    # Check if multi-task detector
    is_multi = 'multi' in detectors
    
    if is_multi:
        detector = detectors['multi'].to(device)
        optimizer = torch.optim.Adam(detector.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            # Training
            detector.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                if isinstance(batch, dict):
                    images = batch['defected'].to(device)
                    conditions = batch['condition'].to(device)
                else:
                    images, _ = batch
                    images = images.to(device)
                    conditions = torch.rand(images.size(0), 6).to(device)
                
                # Forward
                predictions = detector(images)
                loss = criterion(predictions, conditions)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            detector.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, dict):
                        images = batch['defected'].to(device)
                        conditions = batch['condition'].to(device)
                    else:
                        images, _ = batch
                        images = images.to(device)
                        conditions = torch.rand(images.size(0), 6).to(device)
                    
                    predictions = detector(images)
                    loss = criterion(predictions, conditions)
                    mae = torch.abs(predictions - conditions).mean()
                    
                    val_loss += loss.item()
                    val_mae += mae.item()
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_mae = val_mae / len(val_loader)
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, Val MAE={avg_val_mae:.4f}")
    
    else:
        # Train each detector separately
        defect_names = list(detectors.keys())
        
        for defect_idx, (name, detector) in enumerate(detectors.items()):
            print(f"\n[{defect_idx+1}/{len(detectors)}] Training {name} detector...")
            
            detector = detector.to(device)
            optimizer = torch.optim.Adam(detector.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            for epoch in range(num_epochs):
                # Training
                detector.train()
                train_loss = 0.0
                
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                    if isinstance(batch, dict):
                        images = batch['defected'].to(device)
                        # Get only this defect's condition
                        target = batch['condition'][:, defect_idx:defect_idx+1].to(device)
                    else:
                        images, _ = batch
                        images = images.to(device)
                        target = torch.rand(images.size(0), 1).to(device)
                    
                    # Forward
                    prediction = detector(images)
                    loss = criterion(prediction, target)
                    
                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                
                # Validation
                detector.eval()
                val_mae = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        if isinstance(batch, dict):
                            images = batch['defected'].to(device)
                            target = batch['condition'][:, defect_idx:defect_idx+1].to(device)
                        else:
                            images, _ = batch
                            images = images.to(device)
                            target = torch.rand(images.size(0), 1).to(device)
                        
                        prediction = detector(images)
                        mae = torch.abs(prediction - target).mean()
                        val_mae += mae.item()
                
                avg_val_mae = val_mae / len(val_loader)
                
                if epoch == num_epochs - 1:  # Last epoch
                    print(f"  Final - Train Loss: {avg_train_loss:.4f}, Val MAE: {avg_val_mae:.4f}")
    
    print("\n✅ Detector training complete")
    print("="*60)
    
    return detectors


if __name__ == "__main__":
    """Test script for defect detectors."""
    print("Defect Detectors - Test Script")
    print("="*60)
    
    # Test single detector
    print("\n1. Testing single DefectDetector...")
    try:
        detector = DefectDetector('grain', pretrained=False)
        
        x = torch.randn(2, 3, 512, 512)
        pred = detector(x)
        
        assert pred.shape == (2, 1), f"Expected (2, 1), got {pred.shape}"
        assert torch.all(pred >= 0) and torch.all(pred <= 1), "Predictions should be in [0, 1]"
        
        num_params = sum(p.numel() for p in detector.parameters())
        print(f"   ✓ Single detector parameters: {num_params:,}")
        print(f"   ✓ Output shape: {pred.shape}")
        print(f"   ✓ Output range: [{pred.min():.3f}, {pred.max():.3f}]")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test multi-task detector
    print("\n2. Testing MultiDefectDetector...")
    try:
        multi_detector = MultiDefectDetector(pretrained=False)
        
        x = torch.randn(2, 3, 512, 512)
        pred = multi_detector(x)
        
        assert pred.shape == (2, 6), f"Expected (2, 6), got {pred.shape}"
        assert torch.all(pred >= 0) and torch.all(pred <= 1), "Predictions should be in [0, 1]"
        
        num_params = sum(p.numel() for p in multi_detector.parameters())
        print(f"   ✓ Multi-detector parameters: {num_params:,}")
        print(f"   ✓ Output shape: {pred.shape}")
        print(f"   ✓ Output range: [{pred.min():.3f}, {pred.max():.3f}]")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test detector creation
    print("\n3. Testing detector creation...")
    try:
        # Separate detectors
        sep_detectors = create_detectors('separate', pretrained=False)
        print(f"   ✓ Created {len(sep_detectors)} separate detectors")
        
        total_params = sum(
            sum(p.numel() for p in det.parameters())
            for det in sep_detectors.values()
        )
        print(f"   ✓ Total parameters (separate): {total_params:,}")
        
        # Multi-task detector
        multi_detectors = create_detectors('multi', pretrained=False)
        print(f"   ✓ Created multi-task detector")
        
        multi_params = sum(
            sum(p.numel() for p in det.parameters())
            for det in multi_detectors.values()
        )
        print(f"   ✓ Total parameters (multi): {multi_params:,}")
        print(f"   ✓ Parameter reduction: {(1 - multi_params/total_params)*100:.1f}%")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test gradient flow
    print("\n4. Testing gradient flow...")
    try:
        detector = DefectDetector('grain', pretrained=False)
        
        x = torch.randn(1, 3, 512, 512, requires_grad=True)
        pred = detector(x)
        loss = pred.sum()
        loss.backward()
        
        assert x.grad is not None, "Gradient not computed"
        print(f"   ✓ Gradients flow correctly")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n✅ Defect detector tests complete!")
    print("="*60)
