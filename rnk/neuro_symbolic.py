"""
Neuro-Symbolic Refiner

Applies logic/symbolic constraints to guide latent evolution.
Acts as a "corrector" - refines the intent from HRM with
learned constraints.

Design: Lightweight MLP, single-step transformation.
Not a heavy network - just enough to apply corrections.
"""

import torch
import torch.nn as nn
from typing import Optional


class NeuroSymbolicRefiner(nn.Module):
    """
    Neuro-Symbolic refinement layer.
    
    Takes intent from HRM and applies learned corrections
    to ensure logical consistency and constraint satisfaction.
    
    The "symbolic" aspect comes from:
    1. Learned constraint vectors that modulate the latent
    2. Attention to constraint slots
    3. Gated refinement to preserve original intent
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_hidden: int = 512,
        n_constraints: int = 16,  # Learnable constraint vectors
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_constraints = n_constraints
        
        # Learnable constraint embeddings
        self.constraints = nn.Parameter(torch.randn(n_constraints, d_model) * 0.02)
        
        # Constraint attention
        self.constraint_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Refinement MLP
        self.refiner = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        intent: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Apply neuro-symbolic refinement to intent.
        
        Args:
            intent: (batch, n_chunks, d_model) intent from HRM
            return_attention: if True, also return constraint attention weights
            
        Returns:
            refined_intent: (batch, n_chunks, d_model) corrected intent
            attn_weights (optional): (batch, n_chunks, n_constraints)
        """
        batch_size = intent.size(0)
        
        # Expand constraints for batch
        constraints = self.constraints.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Attend to constraints based on intent
        constraint_ctx, attn_weights = self.constraint_attn(
            intent, constraints, constraints,
            need_weights=True
        )
        
        # Apply refinement MLP
        refined = self.refiner(intent + constraint_ctx)
        
        # Gated combination
        gate = self.gate(torch.cat([intent, refined], dim=-1))
        output = gate * refined + (1 - gate) * intent
        
        output = self.layer_norm(output)
        
        if return_attention:
            return output, attn_weights
        return output
    
    def get_constraint_activations(self, intent: torch.Tensor) -> torch.Tensor:
        """
        Get which constraints are being activated for analysis.
        
        Args:
            intent: (batch, n_chunks, d_model)
            
        Returns:
            activations: (batch, n_chunks, n_constraints) constraint attention
        """
        _, attn = self.forward(intent, return_attention=True)
        return attn


class SymbolicRules(nn.Module):
    """
    Optional: Explicit symbolic rule layer.
    
    Learns discrete-ish rules that can be interpreted.
    Uses straight-through estimator for discrete selections.
    
    This is more experimental - use NeuroSymbolicRefiner first.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_rules: int = 32,
        temperature: float = 0.5
    ):
        super().__init__()
        self.n_rules = n_rules
        self.temperature = temperature
        
        # Rule embeddings (condition, action pairs)
        self.rule_conditions = nn.Parameter(torch.randn(n_rules, d_model) * 0.02)
        self.rule_actions = nn.Parameter(torch.randn(n_rules, d_model) * 0.02)
        
        # Rule scoring
        self.rule_scorer = nn.Linear(d_model, n_rules)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply symbolic rules to input.
        
        Args:
            x: (batch, seq, d_model)
        Returns:
            output: (batch, seq, d_model) with rules applied
        """
        # Score each rule
        scores = self.rule_scorer(x)  # (batch, seq, n_rules)
        
        # Soft rule selection (Gumbel-softmax for training)
        if self.training:
            weights = nn.functional.gumbel_softmax(scores, tau=self.temperature, hard=False)
        else:
            weights = nn.functional.softmax(scores / self.temperature, dim=-1)
        
        # Apply weighted action
        actions = torch.einsum('bsn,nd->bsd', weights, self.rule_actions)
        
        # Combine with input
        output = self.out_proj(x + actions)
        
        return output
