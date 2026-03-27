import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Rotary3DPositionEncoding(nn.Module):
    """
    3D Rotary Position Encoding (RoPE) extended to (x, y, z) dimensions.
    
    This extends the standard RoPE from 2D to 3D by applying rotary transformations
    to each spatial dimension independently.
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: int = 10000):
        """
        Initialize 3D RoPE.
        
        Args:
            dim: Dimension of the position encoding (must be divisible by 6 for 3D)
            max_position_embeddings: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Ensure dim is divisible by 6 (3 dimensions * 2 for sin/cos)
        assert dim % 6 == 0, f"dim must be divisible by 6 for 3D RoPE, got {dim}"
        
        # Compute frequency for each dimension
        self.freqs = self._compute_freqs(dim // 6, max_position_embeddings, base)
        
    def _compute_freqs(self, half_dim: int, max_len: int, base: int) -> torch.Tensor:
        """
        Compute frequencies for rotary position encoding.
        
        Args:
            half_dim: Half dimension for each spatial axis
            max_len: Maximum sequence length
            base: Base frequency
            
        Returns:
            Frequency tensor of shape (max_len, half_dim)
        """
        freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, freq)
        return freqs
    
    def forward(self, x: torch.Tensor, coords_3d: torch.Tensor) -> torch.Tensor:
        """
        Apply 3D rotary position encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            coords_3d: 3D coordinates of shape (batch_size, seq_len, 3)
            
        Returns:
            Rotated tensor with 3D position encoding applied
        """
        batch_size, seq_len, dim = x.shape
        
        # Split dimensions for x, y, z axes
        dim_per_axis = dim // 6
        x_x = x[..., 0:dim_per_axis]
        x_y = x[..., dim_per_axis:2*dim_per_axis]
        x_z = x[..., 2*dim_per_axis:3*dim_per_axis]
        x_rest = x[..., 3*dim_per_axis:]
        
        # Get coordinates for each axis
        coords_x = coords_3d[..., 0:1].unsqueeze(-1)  # (batch, seq, 1, 1)
        coords_y = coords_3d[..., 1:2].unsqueeze(-1)
        coords_z = coords_3d[..., 2:3].unsqueeze(-1)
        
        # Compute frequency indices based on coordinates
        freq_x = self._get_freq_for_coord(coords_x, dim_per_axis)
        freq_y = self._get_freq_for_coord(coords_y, dim_per_axis)
        freq_z = self._get_freq_for_coord(coords_z, dim_per_axis)
        
        # Apply rotation to each axis
        x_x_rot = self._rotate(x_x, freq_x)
        x_y_rot = self._rotate(x_y, freq_y)
        x_z_rot = self._rotate(x_z, freq_z)
        
        # Concatenate rotated components
        x_rotated = torch.cat([x_x_rot, x_y_rot, x_z_rot, x_rest], dim=-1)
        
        return x_rotated
    
    def _get_freq_for_coord(self, coord: torch.Tensor, half_dim: int) -> torch.Tensor:
        """
        Get frequency values for given coordinates.
        
        Args:
            coord: Coordinate values of shape (batch, seq, 1, 1)
            half_dim: Half dimension for this axis
            
        Returns:
            Frequency tensor of shape (batch, seq, half_dim)
        """
        # Scale coordinates to frequency indices
        coord_scaled = coord.clamp(0, self.max_position_embeddings - 1).long()
        freqs = self.freqs[coord_scaled.squeeze(-1)]  # (batch, seq, half_dim)
        return freqs
    
    def _rotate(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation using sin and cos of frequencies.
        
        Args:
            x: Input tensor of shape (batch, seq, half_dim)
            freqs: Frequency tensor of shape (batch, seq, half_dim)
            
        Returns:
            Rotated tensor
        """
        half_dim = x.shape[-1]
        x1 = x[..., :half_dim//2]
        x2 = x[..., half_dim//2:]
        
        sin = freqs.sin()
        cos = freqs.cos()
        
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return x_rotated


class DualCoordinatePE(nn.Module):
    """
    Dual Coordinate System Position Encoding.
    
    Combines ego-centric (self-centered) and allocentric (world-centered) 
    position encodings with a learnable gating mechanism.
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: int = 10000):
        """
        Initialize dual coordinate position encoding.
        
        Args:
            dim: Dimension of the position encoding
            max_position_embeddings: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        self.dim = dim
        
        # Ego-centric and allocentric 3D RoPE
        self.ego_rope = Rotary3DPositionEncoding(dim, max_position_embeddings, base)
        self.allo_rope = Rotary3DPositionEncoding(dim, max_position_embeddings, base)
        
        # Learnable gating mechanism
        self.gate = nn.Linear(dim, dim)
        
    def forward(
        self,
        tokens: torch.Tensor,
        ego_coords_3d: torch.Tensor,
        allo_coords_3d: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply dual coordinate position encoding with weighted fusion.
        
        Args:
            tokens: Input tokens of shape (batch_size, seq_len, dim)
            ego_coords_3d: Ego-centric 3D coordinates of shape (batch_size, seq_len, 3)
            allo_coords_3d: Allocentric 3D coordinates of shape (batch_size, seq_len, 3)
            
        Returns:
            Tokens with dual coordinate PE applied
        """
        # Apply ego-centric RoPE
        ego_pe_tokens = self.ego_rope(tokens, ego_coords_3d)
        
        # Apply allocentric RoPE
        allo_pe_tokens = self.allo_rope(tokens, allo_coords_3d)
        
        # Compute gating weights using sigmoid
        gate_weights = torch.sigmoid(self.gate(tokens))
        
        # Weighted fusion of both coordinate systems
        fused_tokens = gate_weights * ego_pe_tokens + (1 - gate_weights) * allo_pe_tokens
        
        return fused_tokens
