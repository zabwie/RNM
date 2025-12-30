"""
Decoder: Renders latent intent into output tokens.

GRU-based decoder with SINGLE cross-attention at initialization.
This is Option 3: inject encoder info once, then pure recurrence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CrossAttentionDecoder(nn.Module):
    """
    GRU decoder with single cross-attention at initialization.
    
    Strategy (Option 3 per user spec):
        1. Cross-attention ONCE to get initial context from encoder
        2. Initialize GRU hidden state with this context
        3. Pure GRU generation (fast, parallelizable for training)
    
    This injects encoder info without O(T²) per-step attention.
    """
    
    def __init__(
        self,
        vocab_size: int = 2048,
        d_model: int = 256,
        d_hidden: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        chunk_size: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.chunk_size = chunk_size
        self.n_layers = n_layers
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Intent projection
        self.intent_proj = nn.Linear(d_model, d_hidden)
        
        # Single cross-attention for initial context
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Projections for cross-attention
        self.query_proj = nn.Linear(d_model, d_hidden)  # Query from intent
        self.key_value_proj = nn.Linear(d_model, d_hidden)  # K,V from encoder
        
        # Project context to initialize hidden state
        self.context_to_hidden = nn.Linear(d_hidden, d_hidden * n_layers)
        
        # GRU - takes [token_emb, intent, context_summary]
        self.gru = nn.GRU(
            input_size=d_model + d_hidden * 2,  # token + intent + context
            hidden_size=d_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size)
        )
    
    def _get_initial_context(
        self,
        intent: torch.Tensor,
        encoder_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single cross-attention to get context from encoder.
        Returns: (context_summary, hidden_init)
        """
        batch_size = intent.shape[0]
        
        # Query = global intent (mean pooled)
        query = self.query_proj(intent.mean(dim=1, keepdim=True))  # (B, 1, d_hidden)
        
        # Key/Value = encoder tokens
        kv = self.key_value_proj(encoder_output)  # (B, enc_len, d_hidden)
        
        # Cross-attention: ONE attention operation
        context, _ = self.cross_attention(query, kv, kv)  # (B, 1, d_hidden)
        context_summary = context.squeeze(1)  # (B, d_hidden)
        
        # Initialize hidden state from context
        hidden_init = self.context_to_hidden(context_summary)  # (B, d_hidden * n_layers)
        hidden_init = hidden_init.view(batch_size, self.n_layers, self.d_hidden)
        hidden_init = hidden_init.transpose(0, 1).contiguous()  # (n_layers, B, d_hidden)
        
        return context_summary, hidden_init
    
    def forward(
        self,
        intent: torch.Tensor,
        encoder_output: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast teacher forcing with single initial cross-attention.
        """
        batch_size, n_chunks, _ = intent.shape
        seq_len = n_chunks * self.chunk_size
        
        # Get context from encoder (SINGLE attention op)
        context_summary, hidden_init = self._get_initial_context(intent, encoder_output)
        
        # Project intent and expand
        intent_cond = self.intent_proj(intent)
        intent_expanded = intent_cond.unsqueeze(2).expand(
            -1, -1, self.chunk_size, -1
        ).reshape(batch_size, seq_len, self.d_hidden)
        
        # Expand context to all positions
        context_expanded = context_summary.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Get token embeddings
        if target_tokens is not None:
            target_tokens = target_tokens[:, :seq_len]
            token_emb = self.embedding(target_tokens)
        else:
            token_emb = torch.zeros(
                batch_size, seq_len, self.d_model,
                device=intent.device, dtype=intent.dtype
            )
        
        # Concatenate [token, intent, context] - FULLY PARALLEL
        gru_input = torch.cat([token_emb, intent_expanded, context_expanded], dim=-1)
        
        # Single GRU forward pass - FAST
        if hidden is None:
            hidden = hidden_init
        
        output, hidden = self.gru(gru_input, hidden)
        logits = self.out_proj(output)
        
        return logits, hidden
    
    def generate(
        self,
        intent: torch.Tensor,
        encoder_output: torch.Tensor,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        bos_token_id: int = 2
    ) -> torch.Tensor:
        """
        Autoregressive generation with encoder context injected once.
        """
        batch_size, n_chunks, _ = intent.shape
        if max_len is None:
            max_len = n_chunks * self.chunk_size
        
        device = intent.device
        
        # Get context from encoder (SINGLE attention - computed once)
        context_summary, hidden = self._get_initial_context(intent, encoder_output)
        
        # Project intent and expand
        intent_cond = self.intent_proj(intent)
        tokens_from_intent = n_chunks * self.chunk_size
        
        if max_len <= tokens_from_intent:
            intent_expanded = intent_cond.unsqueeze(2).expand(
                -1, -1, self.chunk_size, -1
            ).reshape(batch_size, tokens_from_intent, self.d_hidden)[:, :max_len, :]
        else:
            intent_expanded = intent_cond.unsqueeze(2).expand(
                -1, -1, self.chunk_size, -1
            ).reshape(batch_size, tokens_from_intent, self.d_hidden)
            extra = intent_cond[:, -1:, :].expand(-1, max_len - tokens_from_intent, -1)
            intent_expanded = torch.cat([intent_expanded, extra], dim=1)
        
        # Generate tokens
        tokens = [torch.full((batch_size,), bos_token_id, device=device, dtype=torch.long)]
        
        for t in range(max_len):
            curr_token = tokens[-1]
            token_emb = self.embedding(curr_token).unsqueeze(1)  # (B, 1, d_model)
            
            # Intent and context for this step
            intent_t = intent_expanded[:, t:t+1, :]
            context_t = context_summary.unsqueeze(1)
            
            # GRU input
            gru_input = torch.cat([token_emb, intent_t, context_t], dim=-1)
            
            # GRU step
            output, hidden = self.gru(gru_input, hidden)
            logits = self.out_proj(output).squeeze(1)
            
            # Repetition Penalty
            if repetition_penalty != 1.0:
                current_tokens = torch.stack(tokens, dim=1)
                for i in range(batch_size):
                    unique_tokens = torch.unique(current_tokens[i])
                    logit_vals = logits[i, unique_tokens]
                    logits[i, unique_tokens] = torch.where(
                        logit_vals < 0,
                        logit_vals * repetition_penalty,
                        logit_vals / repetition_penalty
                    )
            
            # Temperature
            logits = logits / temperature
            
            # Top-k
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1:]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            tokens.append(next_token)
        
        return torch.stack(tokens[1:], dim=1)


# Backward compatible Decoder
class Decoder(nn.Module):
    """Original GRU decoder without cross-attention."""
    
    def __init__(
        self,
        vocab_size: int = 2048,
        d_model: int = 256,
        d_hidden: int = 256,
        n_layers: int = 2,
        chunk_size: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.chunk_size = chunk_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.intent_proj = nn.Linear(d_model, d_hidden)
        
        self.gru = nn.GRU(
            input_size=d_model + d_hidden,
            hidden_size=d_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size)
        )
    
    def forward(
        self,
        intent: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        sampling_prob: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_chunks, _ = intent.shape
        seq_len = n_chunks * self.chunk_size
        
        intent_cond = self.intent_proj(intent)
        intent_expanded = intent_cond.unsqueeze(2).expand(
            -1, -1, self.chunk_size, -1
        ).reshape(batch_size, seq_len, self.d_hidden)
        
        if target_tokens is not None:
            target_tokens = target_tokens[:, :seq_len]
            token_emb = self.embedding(target_tokens)
        else:
            token_emb = torch.zeros(
                batch_size, seq_len, self.d_model,
                device=intent.device, dtype=intent.dtype
            )
        
        decoder_input = torch.cat([token_emb, intent_expanded], dim=-1)
        
        if hidden is None:
            hidden = torch.zeros(
                self.n_layers, batch_size, self.d_hidden,
                device=intent.device, dtype=intent.dtype
            )
        
        output, hidden = self.gru(decoder_input, hidden)
        logits = self.out_proj(output)
        
        return logits, hidden
    
    def generate(
        self,
        intent: torch.Tensor,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        bos_token_id: int = 2,
        encoder_output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, n_chunks, _ = intent.shape
        if max_len is None:
            max_len = n_chunks * self.chunk_size
        
        device = intent.device
        intent_cond = self.intent_proj(intent)
        tokens_from_intent = n_chunks * self.chunk_size
        
        if max_len <= tokens_from_intent:
            intent_expanded = intent_cond.unsqueeze(2).expand(
                -1, -1, self.chunk_size, -1
            ).reshape(batch_size, tokens_from_intent, self.d_hidden)[:, :max_len, :]
        else:
            intent_expanded = intent_cond.unsqueeze(2).expand(
                -1, -1, self.chunk_size, -1
            ).reshape(batch_size, tokens_from_intent, self.d_hidden)
            extra = intent_cond[:, -1:, :].expand(-1, max_len - tokens_from_intent, -1)
            intent_expanded = torch.cat([intent_expanded, extra], dim=1)
        
        hidden = torch.zeros(
            self.n_layers, batch_size, self.d_hidden,
            device=device, dtype=intent.dtype
        )
        tokens = [torch.full((batch_size,), bos_token_id, device=device, dtype=torch.long)]
        
        for t in range(max_len):
            curr_token = tokens[-1]
            token_emb = self.embedding(curr_token).unsqueeze(1)
            intent_t = intent_expanded[:, t:t+1, :]
            decoder_input = torch.cat([token_emb, intent_t], dim=-1)
            output, hidden = self.gru(decoder_input, hidden)
            logits = self.out_proj(output).squeeze(1)
            logits = logits / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1:]
                logits[indices_to_remove] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            tokens.append(next_token)
        
        return torch.stack(tokens[1:], dim=1)
