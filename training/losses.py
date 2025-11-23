"""
Loss Functions for VintageGAN Training
Specification Reference: Section 4.1

Implements perceptual loss, pixel loss, adversarial loss, and consistency loss
for the three training phases.

Author: VintageGAN Project
Date: 2024
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    
    Specification Reference: Section 4.1 - Phase 1: Generator Pretraining
    
    Uses VGG16 up to conv4_3 layer to extract features and compute
    MSE loss between generated and target feature representations.
    
    Args:
        feature_layers: List of layer indices to use (default: up to conv4_3)
        use_input_norm: Normalize input to VGG range (default: True)
    """
    
    def __init__(
        self,
        feature_layers: list = None,
        use_input_norm: bool = True
    ):
        super().__init__()
        
        # Load pretrained VGG16 (spec: up to conv4_3 = layer 23)
        # Using new PyTorch 2.0+ weights API
        from torchvision.models import VGG16_Weights
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:23])
        
        # Freeze VGG parameters (spec)
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.features.eval()
        self.use_input_norm = use_input_norm
        
        # VGG normalization parameters (ImageNet stats)
        if use_input_norm:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input from [-1, 1] to VGG range.
        
        Args:
            x: Input tensor in [-1, 1]
        
        Returns:
            Normalized tensor for VGG
        """
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        
        # Normalize with ImageNet stats
        x = (x - self.mean) / self.std
        
        return x
    
    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            generated: Generated images (B, 3, H, W) in [-1, 1]
            target: Target images (B, 3, H, W) in [-1, 1]
        
        Returns:
            Perceptual loss (scalar)
        """
        if self.use_input_norm:
            generated = self.normalize_input(generated)
            target = self.normalize_input(target)
        
        # Extract features
        with torch.no_grad():
            target_features = self.features(target)
        
        gen_features = self.features(generated)
        
        # Compute MSE loss (spec)
        loss = F.mse_loss(gen_features, target_features)
        
        return loss


class PixelLoss(nn.Module):
    """
    Pixel-level L1 loss.
    
    Specification Reference: Section 4.1 - Phase 1
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L1 pixel loss.
        
        Args:
            generated: Generated images (B, 3, H, W)
            target: Target images (B, 3, H, W)
        
        Returns:
            L1 loss (scalar)
        """
        return F.l1_loss(generated, target)


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN training.
    
    Specification Reference: Section 4.1 - Phase 3: GAN Fine-Tuning
    
    Uses binary cross-entropy with logits for numerical stability.
    Supports label smoothing.
    """
    
    def __init__(
        self,
        use_label_smoothing: bool = True,
        real_label_value: float = 0.9,
        fake_label_value: float = 0.1
    ):
        super().__init__()
        
        self.use_label_smoothing = use_label_smoothing
        self.real_label_value = real_label_value  # Spec: 0.9
        self.fake_label_value = fake_label_value  # Spec: 0.1
    
    def forward(
        self,
        prediction: torch.Tensor,
        is_real: bool
    ) -> torch.Tensor:
        """
        Compute adversarial loss.
        
        Args:
            prediction: Discriminator predictions (B, 1, H, W) - raw logits
            is_real: Whether target is real (True) or fake (False)
        
        Returns:
            BCE loss (scalar)
        """
        # Create labels with optional smoothing
        if is_real:
            if self.use_label_smoothing:
                target = torch.ones_like(prediction) * self.real_label_value
            else:
                target = torch.ones_like(prediction)
        else:
            if self.use_label_smoothing:
                target = torch.zeros_like(prediction) + self.fake_label_value
            else:
                target = torch.zeros_like(prediction)
        
        # Binary cross-entropy with logits (spec)
        loss = F.binary_cross_entropy_with_logits(prediction, target)
        
        return loss


class ConsistencyLoss(nn.Module):
    """
    Consistency loss to ensure generated images match requested conditions.
    
    Specification Reference: Section 4.2 - Consistency Loss Implementation
    
    Uses pre-trained defect detector networks to extract condition values
    from generated images and compares with target conditions.
    
    Args:
        detectors: Dictionary of detector networks (one per defect type)
    """
    
    def __init__(self, detectors: Dict[str, nn.Module] = None):
        super().__init__()
        
        self.detectors = detectors if detectors is not None else {}
        
        # Freeze detector parameters
        for detector in self.detectors.values():
            for param in detector.parameters():
                param.requires_grad = False
            detector.eval()
    
    def forward(
        self,
        generated: torch.Tensor,
        target_conditions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss.
        
        Args:
            generated: Generated images (B, 3, H, W) in [-1, 1]
            target_conditions: Target condition vector (B, 6) in [0, 1]
        
        Returns:
            MSE loss between detected and target conditions
        """
        if len(self.detectors) == 0:
            # No detectors loaded, return zero loss
            return torch.tensor(0.0, device=generated.device)
        
        # Extract conditions from generated images
        detected_conditions = []
        
        defect_names = ['grain', 'scratch', 'dust', 'vignette', 'color_shift', 'blur']
        
        for i, defect_name in enumerate(defect_names):
            if defect_name in self.detectors:
                with torch.no_grad():
                    detected = self.detectors[defect_name](generated)
                detected_conditions.append(detected)
            else:
                # If detector not available, use target value (no penalty)
                detected_conditions.append(target_conditions[:, i:i+1])
        
        # Stack detected conditions
        detected_conditions = torch.cat(detected_conditions, dim=1)  # (B, 6)
        
        # Compute MSE loss (spec)
        loss = F.mse_loss(detected_conditions, target_conditions)
        
        return loss


class VintageGANLoss(nn.Module):
    """
    Combined loss for VintageGAN training.
    
    Combines all loss terms with appropriate weights based on training phase.
    
    Specification Reference: Section 4.1
    
    Args:
        perceptual_weight: Weight for perceptual loss (default: 0.1)
        pixel_weight: Weight for pixel loss (default: 1.0)
        adversarial_weight: Weight for adversarial loss (default: 1.0)
        consistency_weight: Weight for consistency loss (default: 0.3)
    """
    
    def __init__(
        self,
        perceptual_weight: float = 0.1,
        pixel_weight: float = 1.0,
        adversarial_weight: float = 1.0,
        consistency_weight: float = 0.3,
        detectors: Dict[str, nn.Module] = None
    ):
        super().__init__()
        
        self.perceptual_weight = perceptual_weight
        self.pixel_weight = pixel_weight
        self.adversarial_weight = adversarial_weight
        self.consistency_weight = consistency_weight
        
        # Initialize loss modules
        self.perceptual_loss = VGGPerceptualLoss()
        self.pixel_loss = PixelLoss()
        self.adversarial_loss = AdversarialLoss()
        self.consistency_loss = ConsistencyLoss(detectors=detectors)
    
    def compute_generator_loss(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        conditions: torch.Tensor,
        discriminator_pred: torch.Tensor = None,
        phase: str = 'pretrain'
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute generator loss based on training phase.
        
        Args:
            generated: Generated images (B, 3, H, W)
            target: Target images (B, 3, H, W)
            conditions: Condition vectors (B, 6)
            discriminator_pred: Discriminator predictions (for GAN phase)
            phase: Training phase ('pretrain' or 'gan')
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses = {}
        
        # Pixel loss (all phases)
        pixel_loss = self.pixel_loss(generated, target)
        losses['pixel'] = pixel_loss.item()
        total_loss = self.pixel_weight * pixel_loss
        
        # Perceptual loss (all phases)
        perceptual_loss = self.perceptual_loss(generated, target)
        losses['perceptual'] = perceptual_loss.item()
        total_loss += self.perceptual_weight * perceptual_loss
        
        if phase == 'gan':
            # Adversarial loss (GAN phase only)
            if discriminator_pred is not None:
                adv_loss = self.adversarial_loss(discriminator_pred, is_real=True)
                losses['adversarial'] = adv_loss.item()
                total_loss += self.adversarial_weight * adv_loss
            
            # Consistency loss (GAN phase only)
            consistency_loss = self.consistency_loss(generated, conditions)
            losses['consistency'] = consistency_loss.item()
            total_loss += self.consistency_weight * consistency_loss
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses
    
    def compute_discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute discriminator loss.
        
        Args:
            real_pred: Predictions on real images (B, 1, H, W)
            fake_pred: Predictions on fake images (B, 1, H, W)
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses = {}
        
        # Loss on real images
        real_loss = self.adversarial_loss(real_pred, is_real=True)
        losses['real'] = real_loss.item()
        
        # Loss on fake images
        fake_loss = self.adversarial_loss(fake_pred, is_real=False)
        losses['fake'] = fake_loss.item()
        
        # Total discriminator loss
        total_loss = real_loss + fake_loss
        losses['total'] = total_loss.item()
        
        return total_loss, losses


if __name__ == "__main__":
    """Test script for loss functions."""
    print("Loss Functions - Test Script")
    print("="*60)
    
    # Test VGG perceptual loss
    print("\n1. Testing VGG Perceptual Loss...")
    try:
        perceptual_loss = VGGPerceptualLoss()
        
        gen = torch.randn(2, 3, 256, 256)
        target = torch.randn(2, 3, 256, 256)
        
        loss = perceptual_loss(gen, target)
        
        print(f"   ✓ Perceptual loss: {loss.item():.6f}")
        assert loss.item() > 0, "Loss should be positive"
        
        # Test gradient flow
        loss.backward()
        print("   ✓ Gradient flow: OK")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test pixel loss
    print("\n2. Testing Pixel Loss...")
    try:
        pixel_loss = PixelLoss()
        
        gen = torch.randn(2, 3, 512, 512)
        target = torch.randn(2, 3, 512, 512)
        
        loss = pixel_loss(gen, target)
        
        print(f"   ✓ Pixel loss: {loss.item():.6f}")
        
        # Identical images should have zero loss
        loss_zero = pixel_loss(gen, gen)
        print(f"   ✓ Identical images loss: {loss_zero.item():.6f}")
        assert loss_zero.item() < 1e-6, "Identical images should have zero loss"
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test adversarial loss
    print("\n3. Testing Adversarial Loss...")
    try:
        adv_loss = AdversarialLoss()
        
        pred = torch.randn(2, 1, 32, 32)
        
        real_loss = adv_loss(pred, is_real=True)
        fake_loss = adv_loss(pred, is_real=False)
        
        print(f"   ✓ Real loss: {real_loss.item():.6f}")
        print(f"   ✓ Fake loss: {fake_loss.item():.6f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test consistency loss
    print("\n4. Testing Consistency Loss...")
    try:
        consistency_loss = ConsistencyLoss()
        
        gen = torch.randn(2, 3, 512, 512)
        conditions = torch.rand(2, 6)
        
        loss = consistency_loss(gen, conditions)
        
        print(f"   ✓ Consistency loss: {loss.item():.6f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test combined loss
    print("\n5. Testing Combined VintageGAN Loss...")
    try:
        combined_loss = VintageGANLoss()
        
        gen = torch.randn(2, 3, 256, 256, requires_grad=True)
        target = torch.randn(2, 3, 256, 256)
        conditions = torch.rand(2, 6)
        
        # Pretrain phase
        total, losses = combined_loss.compute_generator_loss(
            gen, target, conditions, phase='pretrain'
        )
        
        print(f"   ✓ Pretrain phase:")
        for key, val in losses.items():
            print(f"      - {key}: {val:.6f}")
        
        # GAN phase
        disc_pred = torch.randn(2, 1, 32, 32)
        total, losses = combined_loss.compute_generator_loss(
            gen, target, conditions, disc_pred, phase='gan'
        )
        
        print(f"   ✓ GAN phase:")
        for key, val in losses.items():
            print(f"      - {key}: {val:.6f}")
        
        # Test backward
        total.backward()
        print("   ✓ Gradient flow: OK")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Loss function tests complete!")
    print("="*60)
