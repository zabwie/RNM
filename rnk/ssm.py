"""
⚠️ DEPRECATED - DO NOT USE ⚠️

This module contains the OLD State Space implementation (FastState + SlowMemory).
It has been REPLACED by Mamba (see mamba.py).

This file is kept for historical reference only.
For active development, use: from .mamba import MambaSSM

Why deprecated:
- FastState + SlowMemory had coordination failures
- Mamba provides unified selective state space with better grokking
- Mamba is more efficient, easier to use, less complicated and less vulnerable to coordination failures or repeated patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FastState(nn.Module):
    """
    Fast state component using stacked GRUs.
    
    Provides high-frequency updates for local coherence.
    Updated every chunk to maintain sequential flow.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Stacked GRU layers
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process chunk through fast state.
        
        Args:
            x: (batch, n_chunks, d_model) chunk latents
            hidden: (n_layers, batch, d_model) previous hidden state
            
        Returns:
            output: (batch, n_chunks, d_model) processed latents
            hidden: (n_layers, batch, d_model) updated hidden state
        """
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = torch.zeros(
                self.n_layers, batch_size, self.d_model,
                device=x.device, dtype=x.dtype
            )
        
        output, hidden = self.gru(x, hidden)
        output = self.layer_norm(output + x)  # Residual connection
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state."""
        return torch.zeros(self.n_layers, batch_size, self.d_model, device=device)


class SlowMemory(nn.Module):
    """
    Slow memory component using slot-based attention.
    
    Provides long-term context storage and retrieval.
    Uses attention for memory read and gated write.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_slots: int = 8,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots
        
        # Learnable memory slots
        self.memory_slots = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        
        # Attention for reading from memory
        self.read_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Write gate
        self.write_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Write projection
        self.write_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model * 2, d_model)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from and write to slow memory.
        
        Args:
            x: (batch, n_chunks, d_model) input latents
            memory: (batch, n_slots, d_model) memory state, or None to init
            
        Returns:
            output: (batch, n_chunks, d_model) memory-enhanced latents
            memory: (batch, n_slots, d_model) updated memory
        """
        batch_size, n_chunks, _ = x.shape
        
        # Initialize memory if needed
        if memory is None:
            memory = self.memory_slots.unsqueeze(0).expand(batch_size, -1, -1)
        
        outputs = []
        
        for t in range(n_chunks):
            chunk = x[:, t:t+1, :]  # (batch, 1, d_model)
            
            # Read from memory via attention
            read_out, _ = self.read_attn(chunk, memory, memory)
            
            # Combine input with memory read
            combined = torch.cat([chunk, read_out], dim=-1)
            enhanced = self.out_proj(combined)
            
            # Compute write gate
            gate = self.write_gate(combined)  # (batch, 1, d_model)
            
            # Update memory with gated write
            write_content = self.write_proj(chunk)  # (batch, 1, d_model)
            
            # Soft attention-based write (update all slots proportionally)
            write_weights = F.softmax(
                torch.bmm(write_content, memory.transpose(1, 2)) / (self.d_model ** 0.5),
                dim=-1
            )  # (batch, 1, n_slots)
            
            # Update each slot
            update = gate * write_content  # (batch, 1, d_model)
            memory = memory + write_weights.transpose(1, 2) * update
            
            outputs.append(enhanced)
        
        output = torch.cat(outputs, dim=1)
        output = self.layer_norm(output + x)  # Residual
        
        return output, memory
    
    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize memory state."""
        return self.memory_slots.unsqueeze(0).expand(batch_size, -1, -1).clone()


class StateSpaceModule(nn.Module):
    """
    Combined State Space Module with fast and slow components.
    
    Flow:
        input → FastState (local coherence) → SlowMemory (long-term context) → output
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_fast_layers: int = 2,
        n_memory_slots: int = 8,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.fast_state = FastState(d_model, n_fast_layers, dropout)
        self.slow_memory = SlowMemory(d_model, n_memory_slots, n_heads, dropout)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        fast_hidden: Optional[torch.Tensor] = None,
        slow_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process through fast + slow state space.
        
        Args:
            x: (batch, n_chunks, d_model) chunk latents
            fast_hidden: previous fast state
            slow_memory: previous slow memory
            
        Returns:
            output: (batch, n_chunks, d_model) processed latents
            fast_hidden: updated fast hidden state
            slow_memory: updated slow memory
        """
        # Fast state for local coherence
        fast_out, fast_hidden = self.fast_state(x, fast_hidden)
        
        # Slow memory for long-term context
        slow_out, slow_memory = self.slow_memory(fast_out, slow_memory)
        
        # Fuse fast and slow
        fused = self.fusion(torch.cat([fast_out, slow_out], dim=-1))
        output = self.layer_norm(fused + x)
        
        return output, fast_hidden, slow_memory
    
    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize both fast and slow states."""
        fast_hidden = self.fast_state.init_hidden(batch_size, device)
        slow_memory = self.slow_memory.init_memory(batch_size, device)
        return fast_hidden, slow_memory
