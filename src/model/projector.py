"""Spatial Projector module for projecting 3D-aware visual tokens to LLM embedding space."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SpatialProjector(nn.Module):
    """Project 3D-aware visual tokens to LLM embedding space.
    
    Supports two projection methods:
    - MLP: Simple multi-layer perceptron projection
    - Cross-Attention: Cross-attention based projection with LLM embeddings as queries
    """
    
    def __init__(
        self,
        visual_token_dim: int,
        llm_embed_dim: int,
        projection_type: str = "mlp",
        hidden_dim: int = 4096,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """Initialize the spatial projector.
        
        Args:
            visual_token_dim: Dimension of 3D-aware visual tokens
            llm_embed_dim: Dimension of LLM embeddings
            projection_type: Type of projection ('mlp' or 'cross_attention')
            hidden_dim: Hidden dimension for MLP or cross-attention
            num_layers: Number of layers for MLP
            num_heads: Number of attention heads for cross-attention
            dropout: Dropout rate
        """
        super().__init__()
        self.visual_token_dim = visual_token_dim
        self.llm_embed_dim = llm_embed_dim
        self.projection_type = projection_type
        
        if projection_type == "mlp":
            layers = []
            input_dim = visual_token_dim
            for i in range(num_layers):
                if i == num_layers - 1:
                    output_dim = llm_embed_dim
                else:
                    output_dim = hidden_dim
                layers.append(nn.Linear(input_dim, output_dim))
                if i < num_layers - 1:
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
                input_dim = output_dim
            self.mlp = nn.Sequential(*layers)
            
        elif projection_type == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=llm_embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.visual_projection = nn.Linear(visual_token_dim, llm_embed_dim)
            self.query_projection = nn.Linear(llm_embed_dim, llm_embed_dim)
            self.output_projection = nn.Linear(llm_embed_dim, llm_embed_dim)
            
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")
    
    def forward(
        self,
        visual_tokens: torch.Tensor,
        llm_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Project visual tokens to LLM embedding space.
        
        Args:
            visual_tokens: 3D-aware visual tokens of shape (batch_size, num_tokens, visual_token_dim)
            llm_embeddings: Optional LLM embeddings for cross-attention queries
                           of shape (batch_size, num_queries, llm_embed_dim)
        
        Returns:
            Projected tokens of shape (batch_size, num_tokens, llm_embed_dim)
        """
        if self.projection_type == "mlp":
            projected_tokens = self.mlp(visual_tokens)
            
        elif self.projection_type == "cross_attention":
            if llm_embeddings is None:
                raise ValueError("llm_embeddings must be provided for cross-attention projection")
            
            batch_size, num_visual_tokens, _ = visual_tokens.shape
            num_queries = llm_embeddings.shape[1]
            
            # Project visual tokens to LLM embedding dimension
            visual_projected = self.visual_projection(visual_tokens)
            
            # Use LLM embeddings as queries
            queries = self.query_projection(llm_embeddings)
            
            # Cross-attention: queries attend to visual tokens
            attn_output, _ = self.cross_attn(
                query=queries,
                key=visual_projected,
                value=visual_projected
            )
            
            # Output projection
            projected_tokens = self.output_projection(attn_output)
            
        else:
            raise ValueError(f"Unknown projection type: {self.projection_type}")
        
        return projected_tokens
