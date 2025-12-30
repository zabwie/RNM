"""
Mamba State Space Model - Pure PyTorch Implementation

Based on https://github.com/alxndrTL/mamba.py
No CUDA compilation required. Windows compatible.

This replaces FastState + SlowMemory with a single unified Mamba backbone.
"""

import math
from dataclasses import dataclass
from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pscan import pscan


@dataclass
class MambaConfig:
    """Configuration for Mamba model."""
    d_model: int = 256  # D
    n_layers: int = 4   # Number of Mamba layers
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16   # N in paper (state dimension)
    expand_factor: int = 2  # E in paper
    d_conv: int = 4     # Convolution kernel size
    
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    
    rms_norm_eps: float = 1e-5
    bias: bool = False
    conv_bias: bool = True
    
    pscan: bool = True  # Use parallel scan (faster) vs sequential
    
    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    """RMS Normalization."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class MambaBlock(nn.Module):
    """
    Single Mamba block with selective state space.
    
    Input: (B, L, D) -> Output: (B, L, D)
    """
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        # Project input to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)
        
        # Depthwise conv for local context
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1
        )
        
        # Project to delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        
        # Project delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        
        # Initialize dt
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # dt bias initialization
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # S4D real initialization for A
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True
        
        # Output projection
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
    
    def forward(self, x):
        """Forward pass: (B, L, D) -> (B, L, D)"""
        _, L, _ = x.shape
        
        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)
        
        # x branch: conv + silu + ssm
        x = x.transpose(1, 2)  # (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  # Depthwise conv
        x = x.transpose(1, 2)  # (B, L, ED)
        x = F.silu(x)
        y = self.ssm(x)
        
        # z branch: silu gate
        z = F.silu(z)
        
        # Combine and project
        output = y * z
        output = self.out_proj(output)  # (B, L, D)
        
        return output
    
    def ssm(self, x):
        """Selective State Space Model."""
        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()
        
        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1
        )
        
        delta = self.dt_proj.weight @ delta.transpose(1, 2)  # (B, ED, L)
        delta = delta.transpose(1, 2)  # (B, L, ED)
        delta = F.softplus(delta + self.dt_proj.bias)
        
        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)
        
        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        """Parallel selective scan using pscan."""
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)
        BX = deltaB * x.unsqueeze(-1)  # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)
        
        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED)
        y = y + D * x
        
        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        """Sequential selective scan (slower but more stable)."""
        _, L, _ = x.shape
        
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)
        BX = deltaB * x.unsqueeze(-1)  # (B, L, ED, N)
        
        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=x.device)
        hs = []
        
        for t in range(L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
        
        hs = torch.stack(hs, dim=1)  # (B, L, ED, N)
        
        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        y = y + D * x
        
        return y
    
    def step(self, x, cache):
        """Single step for inference. x: (B, D), cache: (h, inputs)"""
        h, inputs = cache
        
        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=1)
        
        # Conv step
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]
        x = F.silu(x)
        
        # SSM step
        y, h = self.ssm_step(x, h)
        
        # Gate and output
        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)
        
        # Update cache
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)
        cache = (h, inputs)
        
        return output, cache
    
    def ssm_step(self, x, h):
        """Single SSM step for inference."""
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        
        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))
        
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)
        BX = deltaB * x.unsqueeze(-1)
        
        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=x.device)
        
        h = deltaA * h + BX
        y = (h @ C.unsqueeze(-1)).squeeze(2)
        y = y + D * x
        
        return y, h


class ResidualBlock(nn.Module):
    """Mamba block with normalization and residual connection."""
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
    
    def forward(self, x):
        return self.mixer(self.norm(x)) + x
    
    def step(self, x, cache):
        output, cache = self.mixer.step(self.norm(x), cache)
        return output + x, cache


class MambaSSM(nn.Module):
    """
    Mamba-based State Space Module.
    
    Replaces FastState + SlowMemory + Fusion with unified Mamba backbone.
    Much simpler, fewer coordination failures, better grokking.
    """
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        expand_factor: int = 2,
        d_conv: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.config = MambaConfig(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv
        )
        
        self.layers = nn.ModuleList([
            ResidualBlock(self.config) for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        fast_hidden: Optional[torch.Tensor] = None,
        slow_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process through Mamba layers.
        
        Args:
            x: (batch, n_chunks, d_model) chunk latents
            fast_hidden: ignored (for API compatibility)
            slow_memory: ignored (for API compatibility)
        
        Returns:
            output: (batch, n_chunks, d_model)
            fast_hidden: None (Mamba has internal state)
            slow_memory: None
        """
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        
        # Return None for hidden states - Mamba handles internally
        return x, None, None
    
    def init_state(self, batch_size: int, device: torch.device):
        """Initialize states for streaming (API compatibility)."""
        return None, None


# Convenience alias for drop-in replacement
StateSpaceModule = MambaSSM
