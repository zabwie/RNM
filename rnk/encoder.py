"""
Chunk Encoder: Converts token IDs to latent representations.

Handles:
- Token embeddings
- Positional encoding  
- Chunk-level pooling to get single latent per chunk
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (non-learnable)."""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class ChunkEncoder(nn.Module):
    """
    Encodes input tokens into latent chunk representations.
    
    Flow:
        token_ids → embedding → positional encoding → chunk pooling → latent
    """
    
    def __init__(
        self,
        vocab_size: int = 2048,
        d_model: int = 256,
        chunk_size: int = 32,
        max_len: int = 512,
        dropout: float = 0.1,
        pooling: str = "mean"  # "mean", "last", or "attention"
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.pooling = pooling
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embed_scale = math.sqrt(d_model)
        
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Attention pooling (optional)
        if pooling == "attention":
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.pool_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        return_token_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Encode tokens into chunk-level latent representations.
        
        Args:
            token_ids: (batch, seq_len) token indices
            return_token_embeddings: if True, also return per-token embeddings
            
        Returns:
            chunk_latents: (batch, n_chunks, d_model)
            token_embeddings (optional): (batch, seq_len, d_model)
        """
        batch_size, seq_len = token_ids.shape
        
        # Embed tokens
        x = self.embedding(token_ids) * self.embed_scale  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        token_embeddings = x
        
        # Chunk the sequence
        n_chunks = seq_len // self.chunk_size
        if n_chunks == 0:
            n_chunks = 1
            # Pad if sequence shorter than chunk_size
            pad_len = self.chunk_size - seq_len
            x = nn.functional.pad(x, (0, 0, 0, pad_len))
        
        # Reshape into chunks: (batch, n_chunks, chunk_size, d_model)
        x = x[:, :n_chunks * self.chunk_size, :].view(
            batch_size, n_chunks, self.chunk_size, self.d_model
        )
        
        # Pool each chunk
        if self.pooling == "mean":
            chunk_latents = x.mean(dim=2)  # (batch, n_chunks, d_model)
        elif self.pooling == "last":
            chunk_latents = x[:, :, -1, :]  # (batch, n_chunks, d_model)
        elif self.pooling == "attention":
            # Use attention pooling
            chunk_latents = []
            query = self.pool_query.expand(batch_size, -1, -1)
            for i in range(n_chunks):
                chunk = x[:, i, :, :]  # (batch, chunk_size, d_model)
                pooled, _ = self.pool_attn(query, chunk, chunk)
                chunk_latents.append(pooled.squeeze(1))
            chunk_latents = torch.stack(chunk_latents, dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        if return_token_embeddings:
            return chunk_latents, token_embeddings
        return chunk_latents
    
    def get_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get per-token embeddings without chunking."""
        x = self.embedding(token_ids) * self.embed_scale
        x = self.pos_encoder(x)
        return self.dropout(x)
