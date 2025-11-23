"""
Self-Attention Module
Specification Reference: Section 2.3

Implements self-attention mechanism for capturing long-range dependencies
in feature maps, particularly useful for scratch patterns and vignetting.

Author: VintageGAN Project
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Self-attention layer for capturing long-range dependencies.
    
    Specification Reference: Section 2.3 - Self-Attention Module
    
    Location: Bottleneck of generator (32×32 feature resolution)
    
    This module implements scaled dot-product attention to allow the network
    to focus on relevant spatial locations when generating defects like
    scratches and vignetting that span the entire image.
    
    Args:
        in_dim: Number of input channels
        reduction_ratio: Ratio for reducing channels in Q/K (default: 8)
    
    Example:
        >>> attn = SelfAttention(512)
        >>> x = torch.randn(2, 512, 32, 32)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([2, 512, 32, 32])
    """
    
    def __init__(self, in_dim: int, reduction_ratio: int = 8):
        super().__init__()
        
        self.in_dim = in_dim
        self.reduction_ratio = reduction_ratio
        
        # Query, Key, Value projections (spec)
        # Q and K use reduced dimensions for efficiency
        self.query = nn.Conv2d(in_dim, in_dim // reduction_ratio, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // reduction_ratio, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        
        # Learnable scaling parameter (spec: gamma starts at 0)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Output tensor (B, C, H, W) with self-attention applied
        """
        B, C, H, W = x.shape
        
        # Generate Query, Key, Value
        query = self.query(x)  # (B, C//8, H, W)
        key = self.key(x)      # (B, C//8, H, W)
        value = self.value(x)  # (B, C, H, W)
        
        # Reshape for attention computation
        # (B, C//8, H, W) -> (B, H*W, C//8)
        query = query.view(B, -1, H*W).permute(0, 2, 1)
        # (B, C//8, H, W) -> (B, C//8, H*W)
        key = key.view(B, -1, H*W)
        # (B, C, H, W) -> (B, C, H*W)
        value = value.view(B, -1, H*W)
        
        # Compute attention scores: Q @ K^T
        # (B, H*W, C//8) @ (B, C//8, H*W) = (B, H*W, H*W)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)  # Normalize across keys
        
        # Apply attention to values: Attention @ V
        # (B, C, H*W) @ (B, H*W, H*W)^T = (B, C, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        
        # Reshape back to spatial dimensions
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable gamma (spec)
        # gamma starts at 0, so initially output = input
        out = self.gamma * out + x
        
        return out


class ChannelAttention(nn.Module):
    """
    Channel attention module (alternative to spatial attention).
    
    Focuses on which feature channels are most relevant rather than
    which spatial locations.
    
    Args:
        in_dim: Number of input channels
        reduction_ratio: Channel reduction ratio
    """
    
    def __init__(self, in_dim: int, reduction_ratio: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_dim // reduction_ratio, in_dim, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of channel attention.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Output tensor (B, C, H, W) with channel attention applied
        """
        # Average and max pooling
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        
        # Apply attention weights to input
        out = x * attention
        
        return out


class SpatialAttention(nn.Module):
    """
    Spatial attention module.
    
    Focuses on which spatial locations are most relevant.
    
    Args:
        kernel_size: Convolution kernel size
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spatial attention.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Output tensor (B, C, H, W) with spatial attention applied
        """
        # Channel-wise max and avg pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension
        combined = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid
        attention = self.sigmoid(self.conv(combined))
        
        # Apply attention weights to input
        out = x * attention
        
        return out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines channel and spatial attention for comprehensive attention.
    
    Args:
        in_dim: Number of input channels
        reduction_ratio: Channel reduction ratio
        kernel_size: Spatial convolution kernel size
    """
    
    def __init__(
        self,
        in_dim: int,
        reduction_ratio: int = 16,
        kernel_size: int = 7
    ):
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_dim, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying channel then spatial attention.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Output tensor (B, C, H, W) with CBAM applied
        """
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


if __name__ == "__main__":
    """Test script for attention modules."""
    print("Attention Modules - Test Script")
    print("="*60)
    
    # Test self-attention
    print("\n1. Testing Self-Attention module...")
    try:
        attn = SelfAttention(in_dim=512)
        x = torch.randn(2, 512, 32, 32)
        out = attn(x)
        
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
        print(f"   ✓ Input shape: {x.shape}")
        print(f"   ✓ Output shape: {out.shape}")
        print(f"   ✓ Gamma parameter: {attn.gamma.item():.6f}")
        print(f"   ✓ Parameters: {sum(p.numel() for p in attn.parameters()):,}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test channel attention
    print("\n2. Testing Channel Attention module...")
    try:
        ch_attn = ChannelAttention(in_dim=256)
        x = torch.randn(2, 256, 64, 64)
        out = ch_attn(x)
        
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
        print(f"   ✓ Input shape: {x.shape}")
        print(f"   ✓ Output shape: {out.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test spatial attention
    print("\n3. Testing Spatial Attention module...")
    try:
        sp_attn = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 256, 64, 64)
        out = sp_attn(x)
        
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
        print(f"   ✓ Input shape: {x.shape}")
        print(f"   ✓ Output shape: {out.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test CBAM
    print("\n4. Testing CBAM module...")
    try:
        cbam = CBAM(in_dim=256)
        x = torch.randn(2, 256, 64, 64)
        out = cbam(x)
        
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
        print(f"   ✓ Input shape: {x.shape}")
        print(f"   ✓ Output shape: {out.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test gradients
    print("\n5. Testing gradient flow...")
    try:
        attn = SelfAttention(512)
        x = torch.randn(1, 512, 32, 32, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None, "Gradient not computed"
        print(f"   ✓ Gradients flow correctly")
        print(f"   ✓ Input grad norm: {x.grad.norm().item():.6f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n✅ Attention module tests complete!")
    print("="*60)
