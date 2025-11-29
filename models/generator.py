"""
U-Net Generator with Conditioning
Specification Reference: Section 2.1

Implements U-Net architecture with ResNet34 encoder, conditioning vector
integration, and self-attention at bottleneck.

Author: VintageGAN Project
Date: 2024
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .attention import SelfAttention


class SpectralNorm(nn.Module):
    """
    Spectral Normalization wrapper for convolutional layers.
    
    Specification Reference: Section 2.1 - Implementation Notes
    "Use spectral normalization on all convolutional layers to stabilize training"
    """
    
    def __init__(self, module: nn.Module, name: str = 'weight', power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        
        if not self._made_params():
            self._make_params()
    
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = F.normalize(torch.mv(torch.t(w.view(height,-1).data), u.data), dim=0, eps=1e-12)
            u.data = F.normalize(torch.mv(w.view(height,-1).data, v.data), dim=0, eps=1e-12)
        
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))
    
    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False
    
    def _make_params(self):
        w = getattr(self.module, self.name)
        
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data, dim=0, eps=1e-12)
        v.data = F.normalize(v.data, dim=0, eps=1e-12)
        w_bar = nn.Parameter(w.data)
        
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)
    
    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class ConditionProjection(nn.Module):
    """
    Condition Vector Projection Module.
    
    Specification Reference: Section 2.1.1 - Conditioning Vector Integration
    
    Projects 6D condition vector through MLP: 6 → 128 → 512 dimensions,
    then reshapes for concatenation at bottleneck.
    
    Args:
        condition_dim: Input condition dimension (default: 6)
        hidden_dim: Hidden layer dimension (default: 128)
        output_dim: Output dimension (default: 512)
        spatial_size: Spatial size for output (default: 32)
    """
    
    def __init__(
        self,
        condition_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 512,
        spatial_size: int = 32
    ):
        super().__init__()
        
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        self.spatial_size = spatial_size
        
        # MLP projection (spec: 6 → 128 → 512)
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, condition_vec: torch.Tensor) -> torch.Tensor:
        """
        Project condition vector and reshape for spatial concatenation.
        
        Args:
            condition_vec: (B, 6) condition tensor
        
        Returns:
            Spatially replicated condition: (B, 512, 32, 32)
        """
        # Project through MLP
        embedded = self.mlp(condition_vec)  # (B, 512)
        
        # Reshape and spatially replicate (spec)
        embedded = embedded.view(-1, self.output_dim, 1, 1)
        embedded = embedded.expand(-1, -1, self.spatial_size, self.spatial_size)
        
        return embedded


class ConvBlock(nn.Module):
    """
    Convolutional block with normalization and activation.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size
        stride: Stride
        padding: Padding
        use_spectral_norm: Whether to use spectral normalization
        use_dropout: Whether to use dropout
        dropout_rate: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_spectral_norm: bool = True,
        use_dropout: bool = False,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        layers = []
        
        # Convolution
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        if use_spectral_norm:
            conv = SpectralNorm(conv)
        layers.append(conv)
        
        # Normalization
        layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        
        # Activation
        layers.append(nn.ReLU(inplace=True))
        
        # Dropout (for decoder blocks)
        if use_dropout:
            layers.append(nn.Dropout2d(dropout_rate))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpConvBlock(nn.Module):
    """
    Upsampling convolutional block for decoder.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        use_spectral_norm: Whether to use spectral normalization
        dropout_rate: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_spectral_norm: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_spectral_norm=use_spectral_norm,
            use_dropout=True,
            dropout_rate=dropout_rate
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x


class Generator(nn.Module):
    """
    U-Net Generator with ResNet34 encoder and conditioning.
    
    Specification Reference: Section 2.1 - Generator Architecture
    
    Architecture:
    - Input: RGB Image (512×512×3) + Conditioning Vector (6D)
    - Encoder: ResNet34 backbone with 4 conv blocks
    - Bottleneck: Self-attention + condition integration
    - Decoder: 4 upconv blocks with skip connections
    - Output: Tanh activation → [-1, 1] range
    
    Args:
        condition_dim: Dimension of conditioning vector (default: 6)
        use_spectral_norm: Use spectral normalization (default: True)
        use_self_attention: Use self-attention at bottleneck (default: True)
        dropout_rate: Dropout rate for decoder (default: 0.3)
    
    Example:
        >>> gen = Generator()
        >>> img = torch.randn(2, 3, 512, 512)
        >>> cond = torch.rand(2, 6)
        >>> out = gen(img, cond)
        >>> out.shape
        torch.Size([2, 3, 512, 512])
    """
    
    def __init__(
        self,
        condition_dim: int = 6,
        use_spectral_norm: bool = True,
        use_self_attention: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.condition_dim = condition_dim
        self.use_self_attention = use_self_attention
        
        # Condition projection module (spec: 6 → 128 → 512)
        self.condition_proj = ConditionProjection(
            condition_dim=condition_dim,
            hidden_dim=128,
            output_dim=512,
            spatial_size=32
        )
        
        # Input: concatenate spatially replicated condition (spec: 3 + 6 = 9 channels)
        self.input_conv = ConvBlock(
            9, 64, kernel_size=7, stride=1, padding=3,
            use_spectral_norm=use_spectral_norm
        )
        
        # Encoder (ResNet34-based, spec: Section 2.1)
        self.enc1 = self._make_encoder_block(64, 64, use_spectral_norm)  # 512->256
        self.enc2 = self._make_encoder_block(64, 128, use_spectral_norm)  # 256->128
        self.enc3 = self._make_encoder_block(128, 256, use_spectral_norm)  # 128->64
        self.enc4 = self._make_encoder_block(256, 512, use_spectral_norm)  # 64->32
        
        # Bottleneck with self-attention (spec: Section 2.3)
        self.bottleneck_conv = ConvBlock(
            512 + 512, 512,  # 512 from encoder + 512 from condition projection
            kernel_size=3, padding=1,
            use_spectral_norm=use_spectral_norm
        )
        
        if use_self_attention:
            self.self_attention = SelfAttention(in_dim=512)
        else:
            self.self_attention = nn.Identity()
        
        # Decoder (spec: Section 2.1)
        self.dec4 = UpConvBlock(512, 256, use_spectral_norm, dropout_rate)  # 32->64
        self.dec3 = UpConvBlock(256 + 256, 128, use_spectral_norm, dropout_rate)  # 64->128 (+ skip)
        self.dec2 = UpConvBlock(128 + 128, 64, use_spectral_norm, dropout_rate)  # 128->256 (+ skip)
        self.dec1 = UpConvBlock(64 + 64, 64, use_spectral_norm, dropout_rate)  # 256->512 (+ skip)
        
        # Output layer (spec: 64 → 3 channels, Tanh activation)
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        # Initialize weights (spec: He initialization for ReLU)
        self._init_weights()
    
    def _make_encoder_block(
        self,
        in_channels: int,
        out_channels: int,
        use_spectral_norm: bool
    ) -> nn.Module:
        """Create encoder block with downsampling."""
        return nn.Sequential(
            ConvBlock(in_channels, out_channels, stride=2, use_spectral_norm=use_spectral_norm),
            ConvBlock(out_channels, out_channels, stride=1, use_spectral_norm=use_spectral_norm)
        )
    
    def _init_weights(self):
        """Initialize network weights (spec: He initialization)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Check if weight exists (may be wrapped by SpectralNorm)
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif hasattr(m, 'weight_bar'):
                    # SpectralNorm stores weight in weight_bar
                    nn.init.kaiming_normal_(m.weight_bar, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        image: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of generator.
        
        Args:
            image: Input image (B, 3, 512, 512) in range [-1, 1]
            condition: Condition vector (B, 6) in range [0, 1]
        
        Returns:
            Generated image (B, 3, 512, 512) in range [-1, 1]
        """
        # Validate inputs
        assert image.shape[1] == 3, f"Expected 3 channels, got {image.shape[1]}"
        assert condition.shape[1] == self.condition_dim, \
            f"Expected {self.condition_dim}D condition, got {condition.shape[1]}"
        
        # Spatially replicate condition and concatenate with image (spec: Section 2.1.1)
        B, C, H, W = image.shape
        cond_spatial = condition.view(B, self.condition_dim, 1, 1).expand(B, self.condition_dim, H, W)
        x = torch.cat([image, cond_spatial], dim=1)  # (B, 9, 512, 512)
        
        # Input convolution
        x = self.input_conv(x)  # (B, 64, 512, 512)
        
        # Encoder with skip connections (spec: Section 2.1)
        enc1 = self.enc1(x)      # (B, 64, 256, 256)
        enc2 = self.enc2(enc1)   # (B, 128, 128, 128)
        enc3 = self.enc3(enc2)   # (B, 256, 64, 64)
        enc4 = self.enc4(enc3)   # (B, 512, 32, 32)
        
        # Bottleneck: Integrate condition via MLP projection (spec: Section 2.1.1)
        cond_proj = self.condition_proj(condition)  # (B, 512, 32, 32)
        bottleneck = torch.cat([enc4, cond_proj], dim=1)  # (B, 1024, 32, 32)
        bottleneck = self.bottleneck_conv(bottleneck)  # (B, 512, 32, 32)
        
        # Self-attention (spec: Section 2.3)
        bottleneck = self.self_attention(bottleneck)  # (B, 512, 32, 32)
        
        # Decoder with skip connections (spec: Section 2.1)
        dec4 = self.dec4(bottleneck)  # (B, 256, 64, 64)
        dec3 = self.dec3(torch.cat([dec4, enc3], dim=1))  # (B, 128, 128, 128)
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))  # (B, 64, 256, 256)
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))  # (B, 64, 512, 512)
        
        # Output (spec: Tanh activation → [-1, 1])
        output = self.output_conv(dec1)  # (B, 3, 512, 512)
        
        return output


if __name__ == "__main__":
    """Test script for Generator."""
    print("Generator - Test Script")
    print("="*60)
    
    print("\n1. Testing Generator initialization...")
    try:
        gen = Generator()
        total_params = sum(p.numel() for p in gen.parameters())
        trainable_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
        
        print(f"   ✓ Generator created successfully")
        print(f"   ✓ Total parameters: {total_params:,}")
        print(f"   ✓ Trainable parameters: {trainable_params:,}")
        print(f"   ✓ Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n2. Testing forward pass...")
    try:
        gen = Generator()
        gen.eval()
        
        img = torch.randn(2, 3, 512, 512)
        cond = torch.rand(2, 6)
        
        with torch.no_grad():
            out = gen(img, cond)
        
        assert out.shape == img.shape, f"Shape mismatch: {out.shape} != {img.shape}"
        assert out.min() >= -1.5 and out.max() <= 1.5, \
            f"Output range [{out.min():.3f}, {out.max():.3f}] outside expected [-1, 1]"
        
        print(f"   ✓ Input shape: {img.shape}")
        print(f"   ✓ Condition shape: {cond.shape}")
        print(f"   ✓ Output shape: {out.shape}")
        print(f"   ✓ Output range: [{out.min():.3f}, {out.max():.3f}]")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. Testing gradient flow...")
    try:
        gen = Generator()
        gen.train()
        
        img = torch.randn(1, 3, 512, 512, requires_grad=True)
        cond = torch.rand(1, 6)
        
        out = gen(img, cond)
        loss = out.sum()
        loss.backward()
        
        assert img.grad is not None, "Gradient not computed"
        print(f"   ✓ Gradients flow correctly")
        print(f"   ✓ Input grad norm: {img.grad.norm().item():.6f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n4. Testing different input sizes...")
    try:
        gen = Generator()
        gen.eval()
        
        # Test batch sizes
        for batch_size in [1, 2, 4]:
            img = torch.randn(batch_size, 3, 512, 512)
            cond = torch.rand(batch_size, 6)
            
            with torch.no_grad():
                out = gen(img, cond)
            
            assert out.shape[0] == batch_size, f"Batch size mismatch"
            print(f"   ✓ Batch size {batch_size}: OK")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n5. Testing condition independence...")
    try:
        gen = Generator()
        gen.eval()
        
        img = torch.randn(1, 3, 512, 512)
        cond1 = torch.zeros(1, 6)
        cond2 = torch.ones(1, 6)
        
        with torch.no_grad():
            out1 = gen(img, cond1)
            out2 = gen(img, cond2)
        
        diff = (out1 - out2).abs().mean().item()
        print(f"   ✓ Output difference with different conditions: {diff:.6f}")
        assert diff > 0.01, "Outputs should differ with different conditions"
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n✅ Generator tests complete!")
    print("="*60)
