import torch
import torch.nn as nn
from typing import Tuple, Optional


class UncertaintyAwareReader(nn.Module):
    """
    Uncertainty-aware reader for spatial belief memory.
    
    This module reads from spatial memory using cross-attention while simultaneously
    estimating the uncertainty of the retrieved information. The uncertainty signal
    helps the model make more informed decisions based on memory reliability.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        uncertainty_dim: int = 64
    ):
        """
        Initialize the uncertainty-aware reader.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
            uncertainty_dim: Dimension of the uncertainty signal
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.uncertainty_dim = uncertainty_dim
        
        # Cross-attention for querying spatial memory
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for attention output
        self.attn_norm = nn.LayerNorm(embedding_dim)
        
        # Feed-forward network for processing retrieved context
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )
        
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embedding_dim, uncertainty_dim),
            nn.ReLU(),
            nn.Linear(uncertainty_dim, uncertainty_dim // 2),
            nn.ReLU(),
            nn.Linear(uncertainty_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Confidence-aware attention scaling
        self.confidence_scale = nn.Parameter(torch.ones(1))
        
    def read(
        self,
        decoder_hidden: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        confidence_weights: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read spatial context from memory with uncertainty estimation.
        
        Args:
            decoder_hidden: Query from decoder of shape (batch_size, query_len, embedding_dim)
            memory_keys: Memory keys of shape (batch_size, memory_len, embedding_dim)
            memory_values: Memory values of shape (batch_size, memory_len, embedding_dim)
            confidence_weights: Optional confidence weights for memory entries
                              of shape (batch_size, memory_len)
            memory_mask: Optional attention mask for memory entries
                        of shape (batch_size, memory_len)
                        
        Returns:
            Tuple of (spatial_context, uncertainty_signal)
            - spatial_context: Retrieved spatial context (batch_size, query_len, embedding_dim)
            - uncertainty_signal: Uncertainty scores (batch_size, query_len, 1)
        """
        batch_size, query_len, _ = decoder_hidden.shape
        memory_len = memory_keys.shape[1]
        
        # Apply confidence weighting to memory keys if provided
        if confidence_weights is not None:
            # Scale memory keys by confidence weights
            confidence_weights_expanded = confidence_weights.unsqueeze(-1)
            memory_keys_scaled = memory_keys * confidence_weights_expanded * self.confidence_scale
        else:
            memory_keys_scaled = memory_keys
            
        # Cross-attention: query decoder hidden against memory keys
        attn_output, attn_weights = self.cross_attn(
            query=decoder_hidden,
            key=memory_keys_scaled,
            value=memory_values,
            key_padding_mask=memory_mask,
            need_weights=True
        )
        
        # Residual connection and layer normalization
        attn_output = self.attn_norm(attn_output + decoder_hidden)
        
        # Feed-forward processing
        ffn_output = self.ffn(attn_output)
        spatial_context = self.ffn_norm(ffn_output + attn_output)
        
        # Estimate uncertainty from the retrieved context
        uncertainty_signal = self.uncertainty_head(spatial_context)
        
        # Adjust uncertainty based on attention distribution
        # Higher attention concentration = lower uncertainty
        if attn_weights is not None:
            attn_entropy = self._compute_attention_entropy(attn_weights)
            uncertainty_signal = uncertainty_signal * attn_entropy.unsqueeze(-1)
            
        return spatial_context, uncertainty_signal
        
    def _compute_attention_entropy(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of attention weights as an uncertainty indicator.
        
        Args:
            attn_weights: Attention weights of shape (batch_size, num_heads, query_len, memory_len)
            
        Returns:
            Normalized entropy values of shape (batch_size, query_len)
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-8
        attn_weights = attn_weights + epsilon
        
        # Compute entropy for each query position
        log_weights = torch.log(attn_weights)
        entropy = -torch.sum(attn_weights * log_weights, dim=-1)  # (batch, num_heads, query_len)
        
        # Average over attention heads
        entropy = entropy.mean(dim=1)  # (batch, query_len)
        
        # Normalize by maximum possible entropy (log of memory length)
        max_entropy = torch.log(torch.tensor(attn_weights.shape[-1], dtype=torch.float32))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
        
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        confidence_weights: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (alias for read method).
        
        Args:
            decoder_hidden: Query from decoder
            memory_keys: Memory keys
            memory_values: Memory values
            confidence_weights: Optional confidence weights
            memory_mask: Optional attention mask
            
        Returns:
            Tuple of (spatial_context, uncertainty_signal)
        """
        return self.read(
            decoder_hidden=decoder_hidden,
            memory_keys=memory_keys,
            memory_values=memory_values,
            confidence_weights=confidence_weights,
            memory_mask=memory_mask
        )
        
    def batch_read(
        self,
        decoder_hidden: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        confidence_weights: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        num_reads: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform multiple reads from memory with uncertainty estimation.
        
        Args:
            decoder_hidden: Query from decoder (batch_size, query_len, embedding_dim)
            memory_keys: Memory keys (batch_size, memory_len, embedding_dim)
            memory_values: Memory values (batch_size, memory_len, embedding_dim)
            confidence_weights: Optional confidence weights (batch_size, memory_len)
            memory_mask: Optional attention mask (batch_size, memory_len)
            num_reads: Number of read operations to perform
            
        Returns:
            Tuple of (spatial_context, uncertainty_signal)
        """
        spatial_contexts = []
        uncertainty_signals = []
        
        for _ in range(num_reads):
            context, uncertainty = self.read(
                decoder_hidden=decoder_hidden,
                memory_keys=memory_keys,
                memory_values=memory_values,
                confidence_weights=confidence_weights,
                memory_mask=memory_mask
            )
            spatial_contexts.append(context)
            uncertainty_signals.append(uncertainty)
            
        # Average over multiple reads
        avg_spatial_context = torch.stack(spatial_contexts, dim=0).mean(dim=0)
        avg_uncertainty_signal = torch.stack(uncertainty_signals, dim=0).mean(dim=0)
        
        return avg_spatial_context, avg_uncertainty_signal
