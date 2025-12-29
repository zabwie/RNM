"""
Decoder: Renders latent intent into output tokens.

GRU-based autoregressive decoder conditioned on the refined
intent from the Neuro-Symbolic module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Decoder(nn.Module):
    """
    GRU-based decoder for token generation.
    
    Conditioned on refined intent from HRM + NS pipeline.
    Generates tokens autoregressively within each chunk.
    """
    
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
        
        # Token embedding for decoder input
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Intent conditioning projection
        self.intent_proj = nn.Linear(d_model, d_hidden)
        
        # GRU decoder
        self.gru = nn.GRU(
            input_size=d_model + d_hidden,  # token emb + intent
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
    
    def forward(
        self,
        intent: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode intent into token logits (teacher forcing mode).
        
        Args:
            intent: (batch, n_chunks, d_model) refined intent
            target_tokens: (batch, seq_len) target tokens for teacher forcing
            hidden: optional initial hidden state
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: final hidden state
        """
        batch_size, n_chunks, _ = intent.shape
        seq_len = n_chunks * self.chunk_size
        
        # Project intent
        intent_cond = self.intent_proj(intent)  # (batch, n_chunks, d_hidden)
        
        # Expand intent to token level
        intent_expanded = intent_cond.unsqueeze(2).expand(
            -1, -1, self.chunk_size, -1
        ).reshape(batch_size, seq_len, self.d_hidden)
        
        if target_tokens is not None:
            # Teacher forcing: use target tokens
            target_tokens = target_tokens[:, :seq_len]  # Truncate if needed
            token_emb = self.embedding(target_tokens)
        else:
            # Start with zeros (for generation, use generate() method)
            token_emb = torch.zeros(
                batch_size, seq_len, self.d_model,
                device=intent.device, dtype=intent.dtype
            )
        
        # Concatenate token embeddings with intent conditioning
        decoder_input = torch.cat([token_emb, intent_expanded], dim=-1)
        
        # Initialize hidden state
        if hidden is None:
            hidden = torch.zeros(
                self.n_layers, batch_size, self.d_hidden,
                device=intent.device, dtype=intent.dtype
            )
        
        # Decode
        output, hidden = self.gru(decoder_input, hidden)
        
        # Project to vocabulary
        logits = self.out_proj(output)
        
        return logits, hidden
    
    def generate(
        self,
        intent: torch.Tensor,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        bos_token_id: int = 2  # <bos>
    ) -> torch.Tensor:
        """
        Autoregressive generation from intent.
        
        Args:
            intent: (batch, n_chunks, d_model) refined intent
            max_len: maximum tokens to generate (default: n_chunks * chunk_size)
            temperature: sampling temperature
            top_k: top-k filtering (0 = disabled)
            top_p: nucleus sampling threshold (1.0 = disabled)
            bos_token_id: beginning of sequence token
            
        Returns:
            tokens: (batch, max_len) generated token IDs
        """
        batch_size, n_chunks, _ = intent.shape
        if max_len is None:
            max_len = n_chunks * self.chunk_size
        
        device = intent.device
        
        # Project intent - we have n_chunks intent vectors
        # For generation, we need to interpolate to max_len positions
        intent_cond = self.intent_proj(intent)  # (batch, n_chunks, d_hidden)
        
        # Compute how many tokens total from chunks
        tokens_from_intent = n_chunks * self.chunk_size
        
        if max_len <= tokens_from_intent:
            # We have enough intent - expand and truncate
            intent_expanded = intent_cond.unsqueeze(2).expand(
                -1, -1, self.chunk_size, -1
            ).reshape(batch_size, tokens_from_intent, self.d_hidden)[:, :max_len, :]
        else:
            # Need to extrapolate - repeat the last intent chunk
            intent_expanded = intent_cond.unsqueeze(2).expand(
                -1, -1, self.chunk_size, -1
            ).reshape(batch_size, tokens_from_intent, self.d_hidden)
            # Repeat last intent for extra tokens
            extra_needed = max_len - tokens_from_intent
            last_intent = intent_cond[:, -1:, :].expand(-1, extra_needed, -1)
            intent_expanded = torch.cat([intent_expanded, last_intent], dim=1)
        
        # Initialize
        hidden = torch.zeros(
            self.n_layers, batch_size, self.d_hidden,
            device=device, dtype=intent.dtype
        )
        tokens = [torch.full((batch_size,), bos_token_id, device=device, dtype=torch.long)]
        
        for t in range(max_len):
            # Get current token embedding
            curr_token = tokens[-1]
            token_emb = self.embedding(curr_token).unsqueeze(1)  # (batch, 1, d_model)
            
            # Get intent for this position
            intent_t = intent_expanded[:, t:t+1, :]  # (batch, 1, d_hidden)
            
            # Concatenate
            decoder_input = torch.cat([token_emb, intent_t], dim=-1)
            
            # Decode one step
            output, hidden = self.gru(decoder_input, hidden)
            logits = self.out_proj(output).squeeze(1)  # (batch, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1:]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
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
        
        return torch.stack(tokens[1:], dim=1)  # Skip BOS
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize decoder hidden state."""
        return torch.zeros(self.n_layers, batch_size, self.d_hidden, device=device)

    def generate_with_context(
        self,
        intent: torch.Tensor,
        context_tokens: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Generate tokens conditioned on both intent AND context tokens.
        
        First "warms up" the GRU hidden state by processing context tokens,
        then generates new tokens from that warmed-up state.
        This ensures the output is contextually bound to the prompt.
        
        Args:
            intent: (batch, n_chunks, d_model) refined intent
            context_tokens: (batch, context_len) the prompt tokens
            max_new_tokens: number of new tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            
        Returns:
            tokens: (batch, max_new_tokens) generated token IDs
        """
        batch_size = intent.shape[0]
        device = intent.device
        context_len = context_tokens.shape[1]
        
        # Project intent
        intent_cond = self.intent_proj(intent)  # (batch, n_chunks, d_hidden)
        
        # Expand intent to cover context + generation tokens
        total_len = context_len + max_new_tokens
        n_chunks = intent.shape[1]
        tokens_from_intent = n_chunks * self.chunk_size
        
        if total_len <= tokens_from_intent:
            intent_expanded = intent_cond.unsqueeze(2).expand(
                -1, -1, self.chunk_size, -1
            ).reshape(batch_size, tokens_from_intent, self.d_hidden)[:, :total_len, :]
        else:
            intent_expanded = intent_cond.unsqueeze(2).expand(
                -1, -1, self.chunk_size, -1
            ).reshape(batch_size, tokens_from_intent, self.d_hidden)
            extra_needed = total_len - tokens_from_intent
            last_intent = intent_cond[:, -1:, :].expand(-1, extra_needed, -1)
            intent_expanded = torch.cat([intent_expanded, last_intent], dim=1)
        
        # Initialize hidden state
        hidden = torch.zeros(
            self.n_layers, batch_size, self.d_hidden,
            device=device, dtype=intent.dtype
        )
        
        # PHASE 1: Warmup on context tokens (teacher forcing)
        for t in range(context_len):
            token = context_tokens[:, t]
            token_emb = self.embedding(token).unsqueeze(1)  # (batch, 1, d_model)
            intent_t = intent_expanded[:, t:t+1, :]  # (batch, 1, d_hidden)
            decoder_input = torch.cat([token_emb, intent_t], dim=-1)
            _, hidden = self.gru(decoder_input, hidden)
        
        # PHASE 2: Generate new tokens autoregressively
        # Start with the last context token
        curr_token = context_tokens[:, -1]
        generated = []
        
        for t in range(max_new_tokens):
            token_emb = self.embedding(curr_token).unsqueeze(1)
            intent_t = intent_expanded[:, context_len + t:context_len + t + 1, :]
            decoder_input = torch.cat([token_emb, intent_t], dim=-1)
            
            output, hidden = self.gru(decoder_input, hidden)
            logits = self.out_proj(output).squeeze(1)  # (batch, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1:]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p filtering
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
            generated.append(next_token)
            curr_token = next_token
        
        return torch.stack(generated, dim=1)
