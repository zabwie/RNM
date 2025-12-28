"""
Hierarchical Reasoning Module (HRM)

Captures abstract patterns, context, and hierarchy.
Outputs higher-level intent/plan for the next generation step.

Design: "Narrow but deep" - multiple layers but controlled width
to prevent compute explosion.
"""

import torch
import torch.nn as nn
from typing import Optional


class HRMLayer(nn.Module):
    """
    Single HRM layer: Feed-forward + LayerNorm + Residual.
    
    Uses GELU activation and pre-norm for stable training.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_hidden: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Pre-norm
        self.norm = nn.LayerNorm(d_model)
        
        # Feed-forward block
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-norm and residual.
        
        Args:
            x: (batch, seq, d_model)
        Returns:
            output: (batch, seq, d_model)
        """
        return x + self.ff(self.norm(x))


class HRM(nn.Module):
    """
    Hierarchical Reasoning Module.
    
    2-3 stacked layers for abstract reasoning and intent planning.
    Takes SSM output and produces refined intent vectors.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_hidden: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
        use_gru: bool = False  # Optional: add GRU between layers
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_gru = use_gru
        
        # Stack of HRM layers
        self.layers = nn.ModuleList([
            HRMLayer(d_model, d_hidden, dropout)
            for _ in range(n_layers)
        ])
        
        # Optional GRU for temporal reasoning between layers
        if use_gru:
            self.gru = nn.GRU(
                d_model, d_model, num_layers=1,
                batch_first=True
            )
        
        # Final projection to intent space
        self.intent_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        gru_hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Hierarchical reasoning over chunk latents.
        
        Args:
            x: (batch, n_chunks, d_model) SSM-processed latents
            gru_hidden: optional GRU hidden state
            
        Returns:
            intent: (batch, n_chunks, d_model) high-level intent vectors
        """
        # Pass through HRM layers
        for layer in self.layers:
            x = layer(x)
        
        # Optional GRU pass for temporal coherence
        if self.use_gru:
            if gru_hidden is None:
                x, _ = self.gru(x)
            else:
                x, _ = self.gru(x, gru_hidden)
        
        # Project to intent space
        intent = self.intent_proj(x)
        
        return intent
    
    def forward_with_planning(
        self,
        x: torch.Tensor,
        n_planning_steps: int = 1
    ) -> torch.Tensor:
        """
        Multi-step planning: refine intent through multiple passes.
        
        This allows the model to "think" through the intent before
        committing to a final representation.
        
        Args:
            x: (batch, n_chunks, d_model) input latents
            n_planning_steps: number of refinement iterations
            
        Returns:
            intent: (batch, n_chunks, d_model) refined intent
        """
        intent = x
        
        for _ in range(n_planning_steps):
            # Pass through layers
            for layer in self.layers:
                intent = layer(intent)
            
            # Project and add residual from original
            intent = self.intent_proj(intent) + x * 0.1  # Soft residual
        
        return intent
