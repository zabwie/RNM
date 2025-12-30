"""
RNK: Recursive Neuro-Knowledge Model

Main model class that orchestrates the full forward pass:
    Input → Encoder → SSM → HRM → NS → Decoder → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from .encoder import ChunkEncoder
from .ssm import StateSpaceModule
from .hrm import HRM
from .neuro_symbolic import NeuroSymbolicRefiner
from .decoder import Decoder, CrossAttentionDecoder


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
        n_fast_layers: int = 3,  # Increased for better state tracking
        n_memory_slots: int = 16,  # Doubled capacity
        # HRM
        n_hrm_layers: int = 3,
        hrm_use_gru: bool = False,
        # NS
        n_constraints: int = 4,
        # Decoder
        n_decoder_layers: int = 2,
        # Intent Head
        n_intents: int = 7,  # Fact, Greeting, Refusal, Question, Opinion, Joke, Math
        # Training
        dropout: float = 0.1,
        n_planning_steps: int = 5,  # Increased to 5 for deep latent sketching
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
        self.n_intents = n_intents
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
        self.stop_head = nn.Linear(config.d_model, 1)
        
        # Intent Head (Global Anchor)
        self.intent_classifier = nn.Linear(config.d_model, config.n_intents)
        self.intent_embedding = nn.Embedding(config.n_intents, config.d_model)
        
        # Answerability Gate (prevents hallucination)
        # Predicts: Can this prompt be answered confidently?
        # 0 = not answerable (refuse/hedge), 1 = answerable (proceed)
        self.answerable_head = nn.Linear(config.d_model, 1)

        # Latent Predictor (Transition Model)
        # Predicts Intent[t+1] from Intent[t]
        # This breaks the auto-encoder bottleneck and allows generative planning
        self.latent_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, config.d_model)
        )
        
        # Relevance Head
        # Predicts: Is this Response relevant to this Prompt?
        # Input: [Prompt_State; Response_State] (concatenated) -> 1 logit
        self.relevance_head = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, 1)
        )
        
        # Decoder with cross-attention to encoder outputs
        self.decoder = CrossAttentionDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            d_hidden=config.d_model,
            n_layers=config.n_decoder_layers,
            n_heads=4,  # Cross-attention heads
            chunk_size=config.chunk_size,
            dropout=config.dropout
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        target_intent: Optional[torch.Tensor] = None,
        target_answerable: Optional[torch.Tensor] = None,
        split_indices: Optional[torch.Tensor] = None,
        fast_hidden: Optional[torch.Tensor] = None,
        slow_memory: Optional[torch.Tensor] = None,
        prev_intent: Optional[torch.Tensor] = None,
        return_states: bool = False,
        sampling_prob: float = 0.0,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional planning and weighted loss.
        
        Args:
            input_ids: (batch, seq_len) input token IDs
            target_ids: (batch, seq_len) target tokens for teacher forcing
            target_intent: (batch,) target intent for classification
            target_answerable: (batch,) target for answerability head
            split_indices: (batch,) indices indicating where prompt ends and response begins
            fast_hidden: previous fast state (for streaming)
            slow_memory: previous slow memory (for streaming)
            prev_intent: previous intent vector (for smoothing)
            return_states: if True, return internal states for analysis
            sampling_prob: probability of using model prediction instead of target (scheduled sampling)
            
        Returns:
            dict with:
                - logits: (batch, seq_len, vocab_size)
                - loss: scalar if target_ids provided
                - fast_hidden, slow_memory, prev_intent: updated states
        """
        batch_size = input_ids.size(0)
        
        # 1. Encode chunks (also get token embeddings for cross-attention)
        chunk_latents, token_embeddings = self.encoder(input_ids, return_token_embeddings=True)
        # chunk_latents: (batch, n_chunks, d_model)
        # token_embeddings: (batch, seq_len, d_model)
        
        # 2. State-space processing
        ssm_out, fast_hidden, slow_memory = self.ssm(
            chunk_latents, fast_hidden, slow_memory
        )
        
        # 3. Intent Conditioning
        # Critical: During training, we must predict intent from PROMPT only (split_indices)
        # Otherwise the model "cheats" by seeing the response in the mean pooling.
        if split_indices is not None:
            # Training: Use state at end of prompt
            batch_idx = torch.arange(batch_size, device=ssm_out.device)
            # Clamp indices
            safe_indices = split_indices.clamp(0, ssm_out.size(1) - 1)
            global_context = ssm_out[batch_idx, safe_indices]
        else:
            # Inference: We only have the prompt, so use global mean or last state
            # Using mean of prompt is robust
            global_context = ssm_out.mean(dim=1)
            
        intent_logits = self.intent_classifier(global_context)
        
        if target_intent is not None:
             cond_idx = target_intent
        else:
             cond_idx = intent_logits.argmax(dim=-1)
             
        intent_vec = self.intent_embedding(cond_idx).unsqueeze(1)
        
        # Selective Boost: Identify Math Intent (ID 6)
        # Create scale factor: 3.0 for Math, 1.0 for others
        # cond_idx shape: (B,)
        scale = torch.ones_like(cond_idx, dtype=torch.float32, device=input_ids.device)
        scale[cond_idx == 6] = 3.0
        scale = scale.view(-1, 1, 1) # (B, 1, 1) to broadcast over d_model and chunk dimensions
        
        ssm_out = ssm_out + scale * intent_vec # Force Injection (Precision Boost)
        
        # 4. Hierarchical reasoning
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
        
        # 5.5. Answerability Gate
        # Use the prompt-only context (global_context already computed from prompt at split_idx)
        # This predicts BEFORE seeing the answer - causal and honest
        answerable_logits = self.answerable_head(global_context)  # (batch, 1)
        
        # 6. Decode - Full Sequence Training with Masked Loss
        # Train decoder on FULL sequence (prompt + response)
        # But compute loss ONLY on response portion (after split_indices)
        # This teaches GRU to properly continue from prompt while keeping Intent guidance
        
        if target_ids is not None:
            # CRITICAL FIX: Decoder cross-attention should ONLY see prompt tokens
            # During training, we have full sequence, but we must mask response tokens
            # Otherwise decoder "cheats" by attending to the answer during training
            # but during inference those tokens aren't available
            
            if split_indices is not None:
                # Create mask for prompt-only tokens
                # token_embeddings shape: (B, seq_len, d_model)
                max_split = split_indices.max().item()
                # Only give decoder the first max_split tokens (prompt)
                prompt_embeddings = token_embeddings[:, :max_split, :]
            else:
                # No split info - use all embeddings
                prompt_embeddings = token_embeddings
            
            # Decoder sees full sequence for teacher forcing
            # But cross-attention only sees PROMPT embeddings
            logits, dec_hidden = self.decoder(
                refined_intent,
                encoder_output=prompt_embeddings,  # Only prompt! Not response!
                target_tokens=target_ids
            )
            
            # Compute loss only on RESPONSE tokens (after split_idx)
            # Create mask: 0 for prompt tokens, 1 for response tokens
            batch_size_loss = target_ids.shape[0]
            seq_len = logits.shape[1]
            
            if split_indices is not None:
                # Create position indices (0, 1, 2, ...)
                positions = torch.arange(seq_len, device=target_ids.device).unsqueeze(0).expand(batch_size_loss, -1)
                
                # split_indices are raw token indices directly
                token_split_indices = split_indices.unsqueeze(1)
                
                # Mask: 1 where position >= split_idx (response), 0 otherwise (prompt)
                response_mask = (positions >= token_split_indices).float()
            else:
                # No split info - compute loss on everything
                response_mask = torch.ones(batch_size_loss, seq_len, device=target_ids.device)
            
            # Standard LM shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = target_ids[:, 1:logits.size(1)].contiguous()
            shift_mask = response_mask[:, 1:logits.size(1)].contiguous()
            
            # Flatten for cross-entropy
            flat_logits = shift_logits.view(-1, self.config.vocab_size)
            flat_targets = shift_targets.view(-1)
            flat_mask = shift_mask.view(-1)
            
            # Compute per-token loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
            per_token_loss = loss_fct(flat_logits, flat_targets)
            
            # Reshape to (B, Seq-1) to compute per-sample mean
            masked_loss = per_token_loss * flat_mask
            masked_loss = masked_loss.view(batch_size_loss, -1)
            mask_reshaped = flat_mask.view(batch_size_loss, -1)
            
            # Per-sample token loss
            token_loss_per_sample = masked_loss.sum(dim=1) / (mask_reshaped.sum(dim=1) + 1e-8)
        else:
            token_loss_per_sample = torch.zeros(batch_size, device=input_ids.device)
            # Dummy for graph?
            logits = torch.zeros(
                batch_size, 1, self.config.vocab_size, 
                device=input_ids.device, requires_grad=True
            )
            
        result = {
            'logits': logits, # This is now logits for NEXT chunk
            'fast_hidden': fast_hidden,
            'slow_memory': slow_memory,
            'prev_intent': prev_intent if self.config.intent_smoothing > 0 else None
        }
        
        if target_ids is not None:
             # Stop Loss (Same as before, based on Intent[t])
             # ...
            
            # Stop Loss
            # Target is 1 for the last valid chunk, 0 otherwise
            # Find lengths
            mask = (target_ids != 0).long()
            lengths = mask.sum(dim=1)
            last_chunk_indices = (lengths - 1) // self.config.chunk_size
            last_chunk_indices = last_chunk_indices.clamp(max=intent.size(1)-1)
            
            stop_targets = torch.zeros(batch_size, intent.size(1), device=logits.device)
            stop_targets.scatter_(1, last_chunk_indices.unsqueeze(1), 1.0)
            
            stop_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            stop_loss = stop_loss_fct(stop_logits.squeeze(2), stop_targets).mean(dim=1) # (B,)
            
            # Total Loss Per Sample
            # Weighted sum: Token + Stop + Intent
            total_loss_per_sample = token_loss_per_sample + 0.5 * stop_loss
            
            if target_intent is not None:
                intent_loss_fct = nn.CrossEntropyLoss(reduction='none')
                intent_loss = intent_loss_fct(intent_logits, target_intent)
                total_loss_per_sample += 0.5 * intent_loss  # Upweighted to fix mode collapse
                result['intent_loss'] = intent_loss.mean()
            
            # Answerability Loss
            if target_answerable is not None:
                answerable_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                answerable_loss = answerable_loss_fct(
                    answerable_logits.squeeze(-1), 
                    target_answerable
                ) # (B,)
                total_loss_per_sample += 0.3 * answerable_loss
                result['answerable_loss'] = answerable_loss.mean()
            
            # Latent Prediction Loss (Transition Model)
            # Predict Intent[t+1] from Intent[t]
            pred_intents = self.latent_predictor(refined_intent)
            
            # Shift: pred[t] should match target[t+1]
            pred_shift = pred_intents[:, :-1, :]
            target_shift = refined_intent[:, 1:, :] # Self-supervised on future real intent
            
            # Mask: Only penalize if target[t+1] is fully within response (or valid sequence)
            # Actually, predictability is good universally.
            # But let's verify dimensions match
            if pred_shift.shape[1] > 0:
                 latent_loss_fct = nn.MSELoss(reduction='none')
                 latent_loss_raw = latent_loss_fct(pred_shift, target_shift.detach()) 
                 # (B, T, D) -> Mean over T, D -> (B,)
                 latent_loss = latent_loss_raw.mean(dim=(1,2))
                 
                 total_loss_per_sample += 1.0 * latent_loss 
                 result['latent_loss'] = latent_loss.mean()
            
            # Finalize Main Loss (Weighted Mean)
            if sample_weights is not None:
                loss = (total_loss_per_sample * sample_weights).mean()
            else:
                loss = total_loss_per_sample.mean()
            
            if split_indices is not None and split_indices.size(0) > 1:
                # Contrastive Loss (InfoNCE)
                # Maximize similarity between Prompt State and Response State for same sample
                batch_idx = torch.arange(batch_size, device=input_ids.device)
                
                # Clamp indices to valid range
                split_indices = split_indices.clamp(0, ssm_out.size(1) - 1)
                
                z_prompt = ssm_out[batch_idx, split_indices]
                z_response = ssm_out[:, -1, :] # Last state (batch, d_model)
                
                # Normalize
                z_prompt = F.normalize(z_prompt, p=2, dim=1)
                z_response = F.normalize(z_response, p=2, dim=1)
                
                # Similarity matrix (batch, batch)
                logits_con = torch.matmul(z_prompt, z_response.t()) / 0.1
                
                con_loss_fct = nn.CrossEntropyLoss()
                contrastive_loss = con_loss_fct(logits_con, batch_idx)
                
                loss += 1.0 * contrastive_loss
                result['contrastive_loss'] = contrastive_loss
                
                # Relevance Head Loss (Binary Classification)
                # Maximize score for (Prompt, Response) pairs, minimize for (Prompt, Random_Response)
                
                # Positive pairs (diagonal)
                pos_input = torch.cat([z_prompt, z_response], dim=1)
                pos_logits = self.relevance_head(pos_input) # (batch, 1)
                
                # Negative pairs (shifted response)
                # We roll responses by 1 to create mismatched pairs
                neg_response = torch.roll(z_response, shifts=1, dims=0)
                neg_input = torch.cat([z_prompt, neg_response], dim=1)
                neg_logits = self.relevance_head(neg_input) # (batch, 1)
                
                # HARD NEGATIVE (Self-Correction)
                # Teach filter to reject model's own current bad predictions
                # Use the predicted intent from Latent Predictor
                # We use the LAST predicted intent which corresponds to the Response state
                pred_response = pred_intents[:, -1, :] 
                hard_neg_input = torch.cat([z_prompt, pred_response.detach()], dim=1)
                hard_neg_logits = self.relevance_head(hard_neg_input)
                
                # Targets
                pos_target = torch.ones_like(pos_logits)
                neg_target = torch.zeros_like(neg_logits)
                hard_neg_target = torch.zeros_like(hard_neg_logits)
                
                rel_loss_fct = nn.BCEWithLogitsLoss()
                rel_loss = (
                    rel_loss_fct(pos_logits, pos_target) + 
                    0.5 * rel_loss_fct(neg_logits, neg_target) +
                    0.5 * rel_loss_fct(hard_neg_logits, hard_neg_target)
                ) / 2
                
                loss += 0.5 * rel_loss # High weight to enforce relevance
                result['relevance_loss'] = rel_loss
                
            result['loss'] = loss
            result['token_loss'] = token_loss_per_sample.mean()
            result['stop_loss'] = stop_loss.mean() if isinstance(stop_loss, torch.Tensor) else stop_loss
            result['answerable_logits'] = answerable_logits
        
        if return_states:
            result['chunk_latents'] = chunk_latents
            result['ssm_out'] = ssm_out
            result['intent'] = intent
            result['refined_intent'] = refined_intent
            result['stop_logits'] = stop_logits
            result['answerable_logits'] = answerable_logits
        
        return result
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        fast_hidden: Optional[torch.Tensor] = None,
        slow_memory: Optional[torch.Tensor] = None,
        prev_intent: Optional[torch.Tensor] = None,
        target_intent: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate text autoregressively chunk-by-chunk Causal Prediction.
        """
        self.eval()
        
        with torch.no_grad():
            # Save original input for decoder warmup (before padding)
            original_input_ids = input_ids
            
            # Pad input to chunk boundary (match training conditions for encoder/SSM)
            seq_len = input_ids.shape[1]
            pad_len = (self.config.chunk_size - seq_len % self.config.chunk_size) % self.config.chunk_size
            if pad_len > 0:
                padding = torch.zeros(input_ids.shape[0], pad_len, dtype=input_ids.dtype, device=input_ids.device)
                input_ids = torch.cat([input_ids, padding], dim=1)
            
            # Encode and process context (uses padded input for proper chunking)
            # Also get token embeddings for cross-attention
            chunk_latents, token_embeddings = self.encoder(input_ids, return_token_embeddings=True)
            ssm_out, fast_hidden, slow_memory = self.ssm(
                chunk_latents, fast_hidden, slow_memory
            )
            
            # Get intent (global conditioning)
            # Use Last State to match Training 'split_idx' behavior (State at prompt boundary)
            global_context = ssm_out[:, -1, :]
            
            if target_intent is not None:
                 cond_idx = target_intent
            else:
                 intent_logits = self.intent_classifier(global_context)
                 cond_idx = intent_logits.argmax(dim=-1)
                 # print(f"DEBUG: Classifier ID: {cond_idx[0].item()}")
            # intent_vec = torch.zeros(1, 1, self.config.d_model, device=input_ids.device)
            intent_vec = self.intent_embedding(cond_idx).unsqueeze(1)
            
            # Selective Boost for Generation
            scale = torch.ones_like(cond_idx, dtype=torch.float32, device=input_ids.device)
            scale[cond_idx == 6] = 3.0
            scale = scale.view(-1, 1, 1)
            
            ssm_out = ssm_out + scale * intent_vec # Force Injection (Precision Boost)
            
            # Process through HRM (memory + context)
            if self.config.n_planning_steps > 1:
                hrm_out = self.hrm.forward_with_planning(
                    ssm_out, self.config.n_planning_steps
                )
            else:
                hrm_out = self.hrm(ssm_out)

            # Use pure HRM output - NO intent conditioning
            # This lets the actual input content drive generation
            intent = hrm_out
            
            # Refinement
            refined_intent = self.ns_refiner(intent)
            
            # Answerability Gate Check
            # If model predicts it can't answer, we return a special signal
            answerable_logit = self.answerable_head(global_context)
            answerable_prob = torch.sigmoid(answerable_logit)
            
            # Generative Transition: Predict Response Intent from Prompt Intent
            # This is the key fix for "Autoencoder Prompts"
            # We predict the intent for the next chunk (Response) based on the last chunk of Prompt
            next_intent = self.latent_predictor(refined_intent[:, -1:, :])
            
            # Generation Loop with Prompt-Aware Context
            # Instead of generating blindly, we warm up decoder on prompt tokens
            # This ensures Q→A binding
            
        generated_tokens = []
        all_stop_probs = []
        
        # Intent Amplification (Hard Anchor)
        # 1. Decode the predicted intent from the predictor output
        # 2. Add the clean intent embedding back with high scale to force Decoder compliance
        
        if isinstance(refined_intent, torch.Tensor):
            # refined_intent shape: (B, seq_len, D) -> want last chunk
            next_intent = self.latent_predictor(refined_intent[:, -1:, :])
        else:
            next_intent = torch.zeros(1, 1, self.config.d_model, device=input_ids.device)

        if target_intent is not None:
             # Force the anchor to match the target
             best_intent_idx = target_intent
             print(f"DEBUG: Forcing Anchor to Target ID: {best_intent_idx[0].item()}")
        else:
             # Detect Intent ID from the predicted vector
             sims = torch.nn.functional.cosine_similarity(
                 next_intent.squeeze(1), 
                 self.intent_embedding.weight.unsqueeze(0), 
                 dim=-1
             )
             best_intent_idx = sims.argmax(dim=-1) # (B,)
             print(f"DEBUG: Prompt '{original_input_ids[0][0]}' -> Predicted Intent ID: {best_intent_idx[0].item()}")
        
        # Interpolate (Safe Anchor)
        
        # COMPETITIVE GENERATION (The "Brain" Selection)
        # 1. Expand to K candidates
        k = 20 # High K to escape Mode Collapse
        # (B, 1, D) -> (B*k, 1, D)
        candidates = next_intent.repeat_interleave(k, dim=0) 
        
        # 2. Add exploration noise to allow distinct candidates
        noise = torch.randn_like(candidates) * 0.5 # High Noise to force diversity
        candidates = candidates + noise
        
        # 3. Score with Context Filter (Relevance Head)
        # NORMALIZE to match training!
        context_norm = F.normalize(global_context, p=2, dim=1)
        candidates_norm = F.normalize(candidates.squeeze(1), p=2, dim=1)
        
        # Context needs expansion: (B, D) -> (B*k, D)
        context_expanded = context_norm.repeat_interleave(k, dim=0)
        
        # Input: [Context; Candidate]
        rel_input = torch.cat([context_expanded, candidates_norm], dim=1)
        scores = self.relevance_head(rel_input).view(input_ids.shape[0], k) # (B, k)
        
        # 4. Winner-Take-All
        best_indices = scores.argmax(dim=1) # (B,)
        
        # Select winning vectors
        # Gather logic: (B, k, D) -> select (B, 1, D) using best_indices
        candidates_view = candidates.view(input_ids.shape[0], k, -1)
        next_intent = candidates_view[torch.arange(input_ids.shape[0]), best_indices].unsqueeze(1)
        
        print(f"DEBUG: Competitive Selection. Top Score: {scores.max().item():.2f}")
        
        # Anchor Polish (Optional: Blend winner with hard intent if needed)
        # For now, trust the winner
        
        # Detecting Intent ID for debugging
        sims = torch.nn.functional.cosine_similarity(
             next_intent.squeeze(1), 
             self.intent_embedding.weight.unsqueeze(0), 
             dim=-1
        )
        best_intent_idx = sims.argmax(dim=-1)
        print(f"DEBUG: Winner Intent ID: {best_intent_idx[0].item()}")
        
        # 3. Generate from Amplified Plan with cross-attention to input
        new_tokens = self.decoder.generate(
            next_intent,
            encoder_output=token_embeddings,  # Cross-attention to input tokens
            max_len=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        generated_tokens.append(new_tokens)
        
        # Stop prob (from best plan)
        stop_logit = self.stop_head(next_intent[:, -1:, :])
        stop_prob = torch.sigmoid(stop_logit)
        all_stop_probs.append(stop_prob)
                
        if not generated_tokens:
            return torch.tensor([], device=input_ids.device), torch.tensor([1.0], device=input_ids.device), answerable_prob

        all_tokens = torch.cat(generated_tokens, dim=1)
        batch_size = input_ids.shape[0]
        if all_stop_probs:
            stop_probs_out = torch.cat(all_stop_probs, dim=1).view(batch_size, -1)
        else:
            stop_probs_out = torch.tensor([0.0], device=input_ids.device).repeat(batch_size, 1)
        return all_tokens, stop_probs_out, answerable_prob
    
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
