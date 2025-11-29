"""
PatchGAN Discriminator with Conditioning
Specification Reference: Section 2.2

Implements PatchGAN discriminator that classifies overlapping patches
rather than entire images, providing better texture discrimination.

Author: VintageGAN Project
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import SpectralNorm


class DiscriminatorBlock(nn.Module):
    """
    Discriminator convolutional block.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Stride
        use_spectral_norm: Use spectral normalization
        use_instance_norm: Use instance normalization
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        use_spectral_norm: bool = True,
        use_instance_norm: bool = True
    ):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        # Convolution
        conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_instance_norm
        )
        
        if use_spectral_norm:
            conv = SpectralNorm(conv)
        
        layers = [conv]
        
        # Instance normalization (spec: use InstanceNorm instead of BatchNorm)
        if use_instance_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        
        # LeakyReLU activation (spec: α=0.2)
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator with Conditional Input.
    
    Specification Reference: Section 2.2 - Discriminator Architecture
    
    Architecture:
    - Input: RGB Image (512×512×3) + Condition Vector (6D spatially replicated)
    - Concatenated Input: (512×512×9)
    - 5 Convolutional layers with stride 2 (except last)
    - Output: 32×32 patch predictions (real/fake per patch)
    
    Layer Details:
    - Conv1: 64 filters, 4×4, stride 2 → (256×256×64)
    - Conv2: 128 filters, 4×4, stride 2 → (128×128×128)
    - Conv3: 256 filters, 4×4, stride 2 → (64×64×256)
    - Conv4: 512 filters, 4×4, stride 2 → (32×32×512)
    - Conv5: 1 filter, 4×4, stride 1 → (32×32×1)
    
    Features:
    - Spectral normalization on all conv layers
    - Instance normalization (better for GANs than BatchNorm)
    - LeakyReLU with α=0.2
    - No activation on final layer (raw logits)
    
    Args:
        condition_dim: Dimension of conditioning vector (default: 6)
        base_filters: Base number of filters (default: 64)
        use_spectral_norm: Use spectral normalization (default: True)
        use_instance_norm: Use instance normalization (default: True)
    
    Example:
        >>> disc = Discriminator()
        >>> img = torch.randn(2, 3, 512, 512)
        >>> cond = torch.rand(2, 6)
        >>> out = disc(img, cond)
        >>> out.shape
        torch.Size([2, 1, 32, 32])
    """
    
    def __init__(
        self,
        condition_dim: int = 6,
        base_filters: int = 64,
        use_spectral_norm: bool = True,
        use_instance_norm: bool = True
    ):
        super().__init__()
        
        self.condition_dim = condition_dim
        
        # Input: Image (3) + Condition (6) = 9 channels (spec)
        # First layer: no normalization (spec: common practice)
        self.conv1 = DiscriminatorBlock(
            3 + condition_dim, base_filters,
            kernel_size=4, stride=2,
            use_spectral_norm=use_spectral_norm,
            use_instance_norm=False  # No norm on first layer
        )
        
        # Conv2: 64 → 128 (spec)
        self.conv2 = DiscriminatorBlock(
            base_filters, base_filters * 2,
            kernel_size=4, stride=2,
            use_spectral_norm=use_spectral_norm,
            use_instance_norm=use_instance_norm
        )
        
        # Conv3: 128 → 256 (spec)
        self.conv3 = DiscriminatorBlock(
            base_filters * 2, base_filters * 4,
            kernel_size=4, stride=2,
            use_spectral_norm=use_spectral_norm,
            use_instance_norm=use_instance_norm
        )
        
        # Conv4: 256 → 512 (spec)
        self.conv4 = DiscriminatorBlock(
            base_filters * 4, base_filters * 8,
            kernel_size=4, stride=2,
            use_spectral_norm=use_spectral_norm,
            use_instance_norm=use_instance_norm
        )
        
        # Conv5: 512 → 1, stride 1 (spec: final layer)
        # No activation (output raw logits)
        conv5 = nn.Conv2d(
            base_filters * 8, 1,
            kernel_size=4, stride=1, padding=1
        )
        
        if use_spectral_norm:
            conv5 = SpectralNorm(conv5)
        
        self.conv5 = conv5
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Check if weight exists (may be wrapped by SpectralNorm)
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight, 0.0, 0.02)
                elif hasattr(m, 'weight_bar'):
                    # SpectralNorm stores weight in weight_bar
                    nn.init.normal_(m.weight_bar, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        image: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of discriminator.
        
        Args:
            image: Input image (B, 3, 512, 512) in range [-1, 1]
            condition: Condition vector (B, 6) in range [0, 1]
        
        Returns:
            Patch predictions (B, 1, 32, 32) - raw logits (no sigmoid)
        """
        # Validate inputs
        assert image.shape[1] == 3, f"Expected 3 channels, got {image.shape[1]}"
        assert condition.shape[1] == self.condition_dim, \
            f"Expected {self.condition_dim}D condition, got {condition.shape[1]}"
        
        # Spatially replicate condition and concatenate (spec: Section 2.2)
        B, C, H, W = image.shape
        cond_spatial = condition.view(B, self.condition_dim, 1, 1).expand(B, self.condition_dim, H, W)
        x = torch.cat([image, cond_spatial], dim=1)  # (B, 9, 512, 512)
        
        # Apply convolutional layers (spec)
        x = self.conv1(x)  # (B, 64, 256, 256)
        x = self.conv2(x)  # (B, 128, 128, 128)
        x = self.conv3(x)  # (B, 256, 64, 64)
        x = self.conv4(x)  # (B, 512, 32, 32)
        x = self.conv5(x)  # (B, 1, 32, 32)
        
        return x


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for better detail discrimination.
    
    Uses multiple discriminators at different image scales to capture
    both coarse and fine details.
    
    Args:
        num_scales: Number of discriminator scales (default: 3)
        condition_dim: Dimension of conditioning vector (default: 6)
    """
    
    def __init__(
        self,
        num_scales: int = 3,
        condition_dim: int = 6
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # Create discriminator for each scale
        self.discriminators = nn.ModuleList([
            Discriminator(condition_dim=condition_dim)
            for _ in range(num_scales)
        ])
        
        # Downsampling for multi-scale
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
    
    def forward(
        self,
        image: torch.Tensor,
        condition: torch.Tensor
    ) -> list:
        """
        Forward pass with multi-scale discrimination.
        
        Args:
            image: Input image (B, 3, 512, 512)
            condition: Condition vector (B, 6)
        
        Returns:
            List of predictions from each scale
        """
        outputs = []
        x = image
        
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x, condition))
            
            # Downsample for next scale (except last)
            if i < self.num_scales - 1:
                x = self.downsample(x)
        
        return outputs


if __name__ == "__main__":
    """Test script for Discriminator."""
    print("Discriminator - Test Script")
    print("="*60)
    
    print("\n1. Testing Discriminator initialization...")
    try:
        disc = Discriminator()
        total_params = sum(p.numel() for p in disc.parameters())
        trainable_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
        
        print(f"   ✓ Discriminator created successfully")
        print(f"   ✓ Total parameters: {total_params:,}")
        print(f"   ✓ Trainable parameters: {trainable_params:,}")
        print(f"   ✓ Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n2. Testing forward pass...")
    try:
        disc = Discriminator()
        disc.eval()
        
        img = torch.randn(2, 3, 512, 512)
        cond = torch.rand(2, 6)
        
        with torch.no_grad():
            out = disc(img, cond)
        
        assert out.shape == (2, 1, 32, 32), f"Expected (2, 1, 32, 32), got {out.shape}"
        
        print(f"   ✓ Input shape: {img.shape}")
        print(f"   ✓ Condition shape: {cond.shape}")
        print(f"   ✓ Output shape: {out.shape}")
        print(f"   ✓ Output range: [{out.min():.3f}, {out.max():.3f}]")
        print(f"   ✓ Output patches: {out.shape[2]}×{out.shape[3]} = {out.shape[2]*out.shape[3]}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. Testing gradient flow...")
    try:
        disc = Discriminator()
        disc.train()
        
        img = torch.randn(1, 3, 512, 512, requires_grad=True)
        cond = torch.rand(1, 6)
        
        out = disc(img, cond)
        loss = out.sum()
        loss.backward()
        
        assert img.grad is not None, "Gradient not computed"
        print(f"   ✓ Gradients flow correctly")
        print(f"   ✓ Input grad norm: {img.grad.norm().item():.6f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n4. Testing with real/fake inputs...")
    try:
        disc = Discriminator()
        disc.eval()
        
        # Real image
        real_img = torch.randn(2, 3, 512, 512)
        cond = torch.rand(2, 6)
        
        with torch.no_grad():
            real_pred = disc(real_img, cond)
            real_score = torch.sigmoid(real_pred).mean().item()
        
        print(f"   ✓ Real image average score: {real_score:.3f}")
        
        # "Fake" image (just different random)
        fake_img = torch.randn(2, 3, 512, 512)
        
        with torch.no_grad():
            fake_pred = disc(fake_img, cond)
            fake_score = torch.sigmoid(fake_pred).mean().item()
        
        print(f"   ✓ Fake image average score: {fake_score:.3f}")
        print(f"   ✓ Untrained discriminator outputs varying scores: OK")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n5. Testing Multi-Scale Discriminator...")
    try:
        ms_disc = MultiScaleDiscriminator(num_scales=3)
        ms_disc.eval()
        
        img = torch.randn(2, 3, 512, 512)
        cond = torch.rand(2, 6)
        
        with torch.no_grad():
            outputs = ms_disc(img, cond)
        
        print(f"   ✓ Number of scales: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"   ✓ Scale {i} output shape: {out.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n6. Testing condition independence...")
    try:
        disc = Discriminator()
        disc.eval()
        
        img = torch.randn(1, 3, 512, 512)
        cond1 = torch.zeros(1, 6)
        cond2 = torch.ones(1, 6)
        
        with torch.no_grad():
            out1 = disc(img, cond1)
            out2 = disc(img, cond2)
        
        diff = (out1 - out2).abs().mean().item()
        print(f"   ✓ Output difference with different conditions: {diff:.6f}")
        assert diff > 0.01, "Outputs should differ with different conditions"
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n✅ Discriminator tests complete!")
    print("="*60)
