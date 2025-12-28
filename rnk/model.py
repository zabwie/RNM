"""
RNK: Recursive Neuro-Knowledge Model

Main model class that orchestrates the full forward pass:
    Input → Encoder → SSM → HRM → NS → Decoder → Output
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any

from .encoder import ChunkEncoder
from .ssm import StateSpaceModule
from .hrm import HRM
from .neuro_symbolic import NeuroSymbolicRefiner
from .decoder import Decoder


class RNKConfig:
    """Configuration for RNK model."""
    
    def __init__(
        self,
        vocab_size: int = 2048,
        d_model: int = 256,
        d_hidden: int = 512,
        chunk_size: int = 32,
        max_len: int = 512,
        # SSM
        n_fast_layers: int = 2,
        n_memory_slots: int = 8,
        # HRM
        n_hrm_layers: int = 3,
        hrm_use_gru: bool = False,
        # NS
        n_constraints: int = 16,
        # Decoder
        n_decoder_layers: int = 2,
        # Training
        dropout: float = 0.1,
        n_planning_steps: int = 3,  # Increased default for better reasoning
        intent_smoothing: float = 0.8  # Smoothing factor (0.0=disabled)
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.chunk_size = chunk_size
        self.max_len = max_len
        self.n_fast_layers = n_fast_layers
        self.n_memory_slots = n_memory_slots
        self.n_hrm_layers = n_hrm_layers
        self.hrm_use_gru = hrm_use_gru
        self.n_constraints = n_constraints
        self.n_decoder_layers = n_decoder_layers
        self.dropout = dropout
        self.n_planning_steps = n_planning_steps
        self.intent_smoothing = intent_smoothing
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RNKConfig':
        return cls(**d)


class RNK(nn.Module):
    """
    Recursive Neuro-Knowledge Model.
    
    A sample-efficient architecture that processes text through:
    1. Chunk encoding (tokens → latent chunks)
    2. State-space processing (fast local + slow long-term memory)
    3. Hierarchical reasoning (abstract pattern extraction)
    4. Neuro-symbolic refinement (constraint correction)
    5. Decoding (latent → tokens)
    
    The key insight: each chunk is "thought through" via HRM+NS
    before decoding, giving coherence through planning rather than
    pure autoregression.
    """
    
    def __init__(self, config: RNKConfig):
        super().__init__()
        self.config = config
        
        # Chunk encoder
        self.encoder = ChunkEncoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            chunk_size=config.chunk_size,
            max_len=config.max_len,
            dropout=config.dropout,
            pooling="mean"
        )
        
        # State-space module (fast + slow)
        self.ssm = StateSpaceModule(
            d_model=config.d_model,
            n_fast_layers=config.n_fast_layers,
            n_memory_slots=config.n_memory_slots,
            n_heads=4,
            dropout=config.dropout
        )
        
        # Hierarchical reasoning module
        self.hrm = HRM(
            d_model=config.d_model,
            d_hidden=config.d_hidden,
            n_layers=config.n_hrm_layers,
            dropout=config.dropout,
            use_gru=config.hrm_use_gru
        )
        
        # Neuro-symbolic refiner
        self.ns_refiner = NeuroSymbolicRefiner(
            d_model=config.d_model,
            d_hidden=config.d_hidden,
            n_constraints=config.n_constraints,
            dropout=config.dropout
        )
        
        # Stop Head (Turn Predictor)
        self.stop_head = nn.Linear(config.d_model, 1)  # Predicts p(end_of_turn) for each chunk
        
        # Decoder
        self.decoder = Decoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            d_hidden=config.d_model,
            n_layers=config.n_decoder_layers,
            chunk_size=config.chunk_size,
            dropout=config.dropout
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        fast_hidden: Optional[torch.Tensor] = None,
        slow_memory: Optional[torch.Tensor] = None,
        prev_intent: Optional[torch.Tensor] = None,
        return_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Full RNK forward pass.
        
        Args:
            input_ids: (batch, seq_len) input token IDs
            target_ids: (batch, seq_len) target tokens for teacher forcing
            fast_hidden: previous fast state (for streaming)
            slow_memory: previous slow memory (for streaming)
            prev_intent: previous intent vector (for smoothing)
            return_states: if True, return internal states for analysis
            
        Returns:
            dict with:
                - logits: (batch, seq_len, vocab_size)
                - loss: scalar if target_ids provided
                - fast_hidden, slow_memory, prev_intent: updated states
        """
        batch_size = input_ids.size(0)
        
        # 1. Encode chunks
        chunk_latents = self.encoder(input_ids)  # (batch, n_chunks, d_model)
        
        # 2. State-space processing
        ssm_out, fast_hidden, slow_memory = self.ssm(
            chunk_latents, fast_hidden, slow_memory
        )
        
        # 3. Hierarchical reasoning
        if self.config.n_planning_steps > 1:
            intent = self.hrm.forward_with_planning(
                ssm_out, self.config.n_planning_steps
            )
        else:
            intent = self.hrm(ssm_out)
        
        # 3.5 Intent Smoothing (reduce semantic jumps)
        if self.config.intent_smoothing > 0:
            if prev_intent is None:
                # Initialize with current intent
                prev_intent = intent[:, 0:1, :]
            
            # Autoregressive smoothing across chunks
            smoothed_intents = []
            curr_prev = prev_intent
            
            for t in range(intent.size(1)):
                current = intent[:, t:t+1, :]
                # s_t = alpha * s_{t-1} + (1-alpha) * x_t
                smoothed = self.config.intent_smoothing * curr_prev + \
                          (1 - self.config.intent_smoothing) * current
                smoothed_intents.append(smoothed)
                curr_prev = smoothed
            
            intent = torch.cat(smoothed_intents, dim=1)
            # Update prev_intent for next step (last chunk smoothed)
            prev_intent = curr_prev
        
        # 4. Neuro-symbolic refinement
        refined_intent = self.ns_refiner(intent)
        
        # 5. Stop Prediction
        stop_logits = self.stop_head(refined_intent)  # (batch, n_chunks, 1)
        
        # 6. Decode
        logits, dec_hidden = self.decoder(refined_intent, target_ids)
        
        result = {
            'logits': logits,
            'fast_hidden': fast_hidden,
            'slow_memory': slow_memory,
            'prev_intent': prev_intent if self.config.intent_smoothing > 0 else None
        }
        
        # Compute loss if targets provided
        if target_ids is not None:
            # Token Loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # Pad ID
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = target_ids[:, 1:logits.size(1)].contiguous()
            token_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1)
            )
            
            # Stop Loss
            # Target is 1 for the last valid chunk, 0 otherwise
            # Find lengths
            mask = (target_ids != 0).long()
            lengths = mask.sum(dim=1)
            last_chunk_indices = (lengths - 1) // self.config.chunk_size
            last_chunk_indices = last_chunk_indices.clamp(max=intent.size(1)-1)
            
            stop_targets = torch.zeros(batch_size, intent.size(1), device=logits.device)
            stop_targets.scatter_(1, last_chunk_indices.unsqueeze(1), 1.0)
            
            stop_loss_fct = nn.BCEWithLogitsLoss()
            stop_loss = stop_loss_fct(stop_logits.squeeze(2), stop_targets)
            
            # Total Loss
            loss = token_loss + 0.1 * stop_loss # Weight stop loss
            
            result['loss'] = loss
            result['token_loss'] = token_loss
            result['stop_loss'] = stop_loss
        
        if return_states:
            result['chunk_latents'] = chunk_latents
            result['ssm_out'] = ssm_out
            result['intent'] = intent
            result['refined_intent'] = refined_intent
            result['stop_logits'] = stop_logits
        
        return result
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        fast_hidden: Optional[torch.Tensor] = None,
        slow_memory: Optional[torch.Tensor] = None,
        prev_intent: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tokens from input context.
        
        Args:
            input_ids: (batch, seq_len) input context
            max_new_tokens: tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            
        Returns:
            generated_ids: (batch, max_new_tokens)
            stop_probs: (batch, n_chunks) probabilities of stopping for each chunk
        """
        self.eval()
        
        with torch.no_grad():
            # Encode and process context
            chunk_latents = self.encoder(input_ids)
            ssm_out, fast_hidden, slow_memory = self.ssm(
                chunk_latents, fast_hidden, slow_memory
            )
            
            # Get intent
            if self.config.n_planning_steps > 1:
                intent = self.hrm.forward_with_planning(
                    ssm_out, self.config.n_planning_steps
                )
            else:
                intent = self.hrm(ssm_out)
            
            # Smoothing during generation (if using cache)
            if self.config.intent_smoothing > 0 and prev_intent is not None:
                # Only smooth the LAST chunk intent if we are streaming
                # But here we are processing full context + new gen
                # For simplicity in this method, we assume standard generation from context
                pass # Logic already handled in forward if we used it, but here we just used components directly
                
                # Apply smoothing to the sequence we just computed
                if prev_intent is None:
                    prev_intent = intent[:, 0:1, :]
                
                smoothed_intents = []
                curr_prev = prev_intent
                for t in range(intent.size(1)):
                    current = intent[:, t:t+1, :]
                    smoothed = self.config.intent_smoothing * curr_prev + \
                              (1 - self.config.intent_smoothing) * current
                    smoothed_intents.append(smoothed)
                    curr_prev = smoothed
                
                intent = torch.cat(smoothed_intents, dim=1)
            
            # Refine
            refined_intent = self.ns_refiner(intent)
            
            # Check stop probability of the last generated chunk
            stop_logits = self.stop_head(refined_intent)
            stop_probs = torch.sigmoid(stop_logits).squeeze(2) # (batch, n_chunks)
            
            # Generate
            # Note: Decoder now handles arbitrary lengths via interpolation
            generated_ids = self.decoder.generate(
                refined_intent,
                max_len=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            return generated_ids, stop_probs
    
    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize SSM states for streaming generation."""
        return self.ssm.init_state(batch_size, device)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_param_breakdown(self) -> Dict[str, int]:
        """Get parameter count per module."""
        return {
            'encoder': sum(p.numel() for p in self.encoder.parameters()),
            'ssm': sum(p.numel() for p in self.ssm.parameters()),
            'hrm': sum(p.numel() for p in self.hrm.parameters()),
            'ns_refiner': sum(p.numel() for p in self.ns_refiner.parameters()),
            'decoder': sum(p.numel() for p in self.decoder.parameters()),
        }


def create_rnk_small() -> RNK:
    """Create small RNK model (~5M params)."""
    config = RNKConfig(
        vocab_size=2048,
        d_model=192,
        d_hidden=384,
        n_hrm_layers=2,
        n_fast_layers=1,
        n_memory_slots=4,
        n_decoder_layers=1
    )
    return RNK(config)


def create_rnk_base() -> RNK:
    """Create base RNK model (~10M params)."""
    config = RNKConfig(
        vocab_size=2048,
        d_model=256,
        d_hidden=512,
        n_hrm_layers=3,
        n_fast_layers=2,
        n_memory_slots=8,
        n_decoder_layers=2
    )
    return RNK(config)


def create_rnk_large() -> RNK:
    """Create large RNK model (~20M params)."""
    config = RNKConfig(
        vocab_size=2048,
        d_model=384,
        d_hidden=768,
        n_hrm_layers=4,
        n_fast_layers=2,
        n_memory_slots=16,
        n_decoder_layers=2
    )
    return RNK(config)
