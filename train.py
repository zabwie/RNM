"""
RNK Training Script

Training loop with:
- Chunk-level cross-entropy loss
- AdamW optimizer with cosine LR schedule
- Gradient clipping
- Checkpoint saving
- Validation loop
- Curriculum learning support
"""

import os
import json
import math
import time
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Intent Definitions
INTENT_MAP = {
    'greeting': 0,
    'question': 1,
    'fact': 2,  # Default
    'refusal': 3,
    'opinion': 4,
    'joke': 5,
    'math': 6
}

def classify_intent(text: str, prompt: str = "") -> int:
    """Rigorous regex-based intent classification."""
    text = text.lower().strip()
    prompt = prompt.lower().strip()
    
    # 1. Math (Highest Priority) - Fixes "Fact" overlap
    # Detects numbers with operators or math keywords
    if re.search(r'\b(\d+[\s]*[\+\-\*\/][\s]*\d+)\b', text) or \
       re.search(r'\b(sum|difference|product|divided by|plus|minus|calculate)\b', text):
        return INTENT_MAP['math']

    # 2. Joke (Highest Priority Override via Prompt)
    if re.search(r'\b(joke|funny|laugh|pun)\b', prompt):
        return INTENT_MAP['joke']
        
    # 3. Refusal (Negative sentiment)
    if re.search(r'\b(no|not|cannot|can\'t|don\'t|stop|sorry|unable)\b', text):
        return INTENT_MAP['refusal']

    # 4. Greeting (Start of sentence/conversation)
    # Strict anchor at start ^
    if re.search(r'^(hello|hi|hey|greetings|good morning|good evening)\b', text):
        return INTENT_MAP['greeting']
    
    # 5. Question (Ends with ? or starts with WH-word)
    if "?" in text or re.search(r'^(what|why|how|when|where|who)\b', text):
        return INTENT_MAP['question']
        
    # 6. Opinion (Subjective markers)
    if re.search(r'\b(i think|i feel|believe|opinion|my view)\b', text):
        return INTENT_MAP['opinion']
        
    # 7. Fact (Default)
    return INTENT_MAP['fact']

def classify_answerability(prompt: str, response: str) -> float:
    """
    Heuristic answerability classification.
    
    Returns:
        1.0 = answerable (factual, direct, concrete)
        0.0 = not answerable / should refuse
    """
    prompt = prompt.lower()
    response = response.lower()
    
    # Unanswerable patterns (should refuse)
    unanswerable_keywords = [
        "i don't know", "i'm not sure", "uncertain",
        "sorry", "apologize", "unfortunately",
        "cannot", "can't help", "unable to"
    ]
    if any(kw in response for kw in unanswerable_keywords):
        return 0.0
    
    # Answerable patterns (factual questions with short direct answers)
    if "?" in prompt:
        # Check for direct answer length (short = likely answerable)
        response_words = len(response.split())
        if response_words < 50:  # Short, direct answer
            return 1.0
    
    # Greetings are "answerable" (model can greet back)
    greeting_words = ["hi", "hello", "hey", "good morning", "good evening"]
    if any(g in prompt for g in greeting_words):
        return 1.0
    
    # Default: treat as answerable (most chat is)
    return 1.0

def collate_fn(batch):
    """Custom collate function for mixed data types."""
    if len(batch) == 0:
        return torch.tensor([])
        
    elem = batch[0]
    if isinstance(elem, dict):
        # Dict batch (ConversationDataset)
        input_ids = [b['input_ids'] for b in batch]
        intents = torch.stack([b['intent'] for b in batch])
        split_indices = torch.stack([b['split_idx'] for b in batch])
        answerabilities = torch.stack([b['answerable'] for b in batch])
        
        # Weights (Optional)
        weights = None
        if 'weight' in batch[0] and batch[0]['weight'] is not None:
            weights = torch.stack([b['weight'] for b in batch])
        
        # STATIC PADDING: Always pad to fixed length (multiple of 32)
        # This ensures tensor cores are used efficiently
        FIXED_SEQ_LEN = 256  # Must be multiple of 32
        max_len = FIXED_SEQ_LEN
        
        # Critical: Fixed length = fast tensor core kernels 
        # leading to 0 loss and no training.
        chunk_size = 32 
        if max_len % chunk_size != 0:
            max_len = ((max_len // chunk_size) + 1) * chunk_size
            
        padded_ids = torch.zeros(len(input_ids), max_len, dtype=torch.long)
        for i, ids in enumerate(input_ids):
            padded_ids[i, :len(ids)] = ids
            
        return {
            'input_ids': padded_ids.to(intents.device), 
            'intent': intents,
            'split_idx': split_indices,
            'answerable': answerabilities,
            'weight': weights
        }
    elif isinstance(elem, torch.Tensor):
        # Tensor batch (TextDataset)
        # Pad sequences manually if variable length, or use default logical alignment
        # TextDataset already pads to max_len inside __init__?
        # TextDataset.__init__ does pad, but to `max_len`. 
        # But if we want dynamic batching, we should pad here.
        # But TextDataset produced fixed length tensors?
        # Line 81: `tokenizer.encode(..., padding=True)` pads to `max_length=max_len`?
        # Yes. So Tensors are same size.
        return torch.stack(batch)
    
    return torch.utils.data.dataloader.default_collate(batch)

from rnk import RNK
from rnk.model import RNKConfig
from rnk.byte_encoder import ByteEncoder  # Raw byte encoding (no tokenization)


@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    train_data_path: str = "data/train.txt"
    val_data_path: Optional[str] = "data/val.txt"
    tokenizer_path: str = "data/tokenizer.json"
    max_samples: Optional[int] = 100000  # Increased to 100k
    
    # Model
    model_size: str = "base"  # "small", "base", "large"
    
    # Training
    batch_size: int = 512  # Optimized for GPU throughput
    epochs: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100
    
    # Sequence length - Longer for raw byte input (no tokenization)
    use_curriculum: bool = False  # Disabled for static shapes
    initial_seq_len: int = 1024   # Longer for raw byte sequences
    max_seq_len: int = 1024       # Raw bytes need more length
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1  # epochs
    
    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # Resume
    resume_path: Optional[str] = None


class TextDataset(Dataset):
    """Simple text dataset for RNK training."""
    
    def __init__(
        self,
        texts: List[str],
        encoder: ByteEncoder,  # Raw byte encoder (no tokenization)
        max_len: int = 1024
    ):
        self.encoder = encoder
        self.max_len = max_len
        
        # Encode all texts to raw bytes
        self.samples = []
        for text in texts:
            ids = encoder.encode(text, max_length=max_len, padding=True)
            if len(ids) >= 16:  # Skip very short samples
                self.samples.append(ids)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.samples[idx], dtype=torch.long)


class ConversationDataset(Dataset):
    """Dataset for conversational pairs (input → response)."""
    
    def __init__(
        self,
        conversations: List[Tuple[str, str]],
        encoder: ByteEncoder,  # Raw byte encoder (no tokenization)
        max_len: int = 1024
    ):
        self.encoder = encoder
        self.max_len = max_len
        
        self.encodings = []
        self.intents = []
        self.split_indices = []
        self.answerabilities = []  # Answerability Gate labels
        self.weights = []
        
        for item in conversations:
            weight = 1.0
            inp, out = None, None
            weight = 1.0
            is_pair = False
            
            if isinstance(item, dict):
                inp = item['input']
                out = item['output']
                weight = item.get('weight', 1.0)
                is_pair = True
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                inp, out = item
                is_pair = True
            
            if is_pair:
                # Format: "User: <inp>\nModel: <out>"
                prompt_text = f"User: {inp}\nModel:"
                full_text = f"{prompt_text} {out}"
                
                intent = classify_intent(inp, prompt=inp)
                answerable = classify_answerability(inp, out)
                
                # Track prompt end (Prediction Point)
                prompt_enc = self.encoder.encode(prompt_text, padding=False)
                
                # Boundary Padding: Pad prompt to chunk boundary (32)
                chunk_size = 32 # Default RNK chunk size
                pad_len = (chunk_size - len(prompt_enc) % chunk_size) % chunk_size
                if pad_len > 0:
                    prompt_enc = prompt_enc + [0] * pad_len
                
                split_idx = len(prompt_enc)
                
                # Encode response and combine
                response_enc = self.encoder.encode(f" {out}", padding=False)
                ids = prompt_enc + response_enc
            else:
                full_text = str(item)
                intent = INTENT_MAP['fact']
                answerable = 1.0  # Default answerable
                split_idx = 0
                ids = self.encoder.encode(full_text, max_length=self.max_len, padding=False)
            # Wait, `collate_fn` handles padding now.
            # But line 108 had `padding=True`.
            # If we switch to dynamic padding, we save memory.
            # Let's keep `padding=True` logic for consistency unless I changed TextDataset.
            # I can't change TextDataset easily in replace_file.
            # So I will just stick to fixed padding to `max_len` logic if possible, or use collate pad.
            # My `collate_fn` implements padding.
            # So I should disable padding here.
            
            if len(ids) >= 16:
                if len(ids) > max_len:
                    ids = ids[:max_len]
                    if split_idx >= max_len: split_idx = max_len - 1
                self.encodings.append(torch.tensor(ids, dtype=torch.long))
                self.intents.append(torch.tensor(intent, dtype=torch.long))
                self.split_indices.append(torch.tensor(split_idx, dtype=torch.long))
                self.answerabilities.append(torch.tensor(answerable, dtype=torch.float))
                self.weights.append(torch.tensor(weight, dtype=torch.float))
    
    def __len__(self) -> int:
        return len(self.encodings)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': self.encodings[idx],
            'intent': self.intents[idx],
            'split_idx': self.split_indices[idx],
            'answerable': self.answerabilities[idx],
            'weight': self.weights[idx]
        }


def load_texts_from_file(path: str) -> List[str]:
    """Load texts from file (one per line)."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_conversations_from_jsonl(path: str) -> List[Tuple[str, str]]:
    """Load conversations from JSONL - auto-detects format.
    
    Supports:
    - Simple format: {"input": ..., "output": ...}
    - OASST2 format: {"text": ..., "role": ..., "parent_id": ...}
    """
    conversations = []
    messages = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            messages.append(data)
    
    # Detect format
    if messages and ('input' in messages[0] or 'instruction' in messages[0]):
        # Simple format (Dolly/Alpaca style)
        for data in messages:
            # Normalize keys
            if 'instruction' in data and 'input' not in data:
                # If context exists, append it? Or just use instruction?
                # Usually instruction + context
                if data.get('context'):
                    data['input'] = f"{data['instruction']}\n{data['context']}"
                else:
                    data['input'] = data['instruction']
            
            if 'response' in data and 'output' not in data:
                data['output'] = data['response']
                
            # Pass full dict to support 'weight'
            conversations.append(data)
    elif messages and 'role' in messages[0]:
        # OASST2 format - build parent-child pairs
        msg_dict = {m['message_id']: m for m in messages}
        
        for msg in messages:
            # Find assistant replies to user messages
            if msg['role'] == 'assistant' and msg.get('parent_id'):
                parent = msg_dict.get(msg['parent_id'])
                if parent and parent.get('role') == 'prompter':
                    user_text = parent.get('text', '')
                    assistant_text = msg.get('text', '')
                    if user_text and assistant_text:
                        conversations.append((user_text, assistant_text))
    
    return conversations


class Trainer:
    """RNK Trainer."""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        
        # Device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = self._init_tokenizer()
        
        # Create model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Resume
        if self.config.resume_path:
            if os.path.exists(self.config.resume_path):
                print(f"Resuming training from {self.config.resume_path}...")
                checkpoint = torch.load(self.config.resume_path, map_location=self.device)
                state_dict = checkpoint['model_state']
                
                # Check for Intent Embedding Mismatch (6 -> 7)
                if 'intent_embedding.weight' in state_dict:
                    emb = state_dict['intent_embedding.weight']
                    if emb.shape[0] == 6 and self.model.config.n_intents == 7:
                        print("Patching Intent Embedding (6 -> 7) for Math Intent...")
                        # Create new embedding tensor (7, D)
                        new_emb = torch.zeros(7, emb.shape[1], device=emb.device)
                        new_emb[:6] = emb 
                        new_emb[6] = emb[2].clone() + torch.randn_like(emb[2]) * 0.02
                        state_dict['intent_embedding.weight'] = new_emb
                
                # Check for Intent Classifier Mismatch (6 -> 7)
                if 'intent_classifier.weight' in state_dict:
                     cls_w = state_dict['intent_classifier.weight']
                     if cls_w.shape[0] == 6 and self.model.config.n_intents == 7:
                         print("Patching Intent Classifier (6 -> 7)...")
                         # Weight: (Out, In) -> (7, 256)
                         new_w = torch.zeros(7, cls_w.shape[1], device=cls_w.device)
                         new_w[:6] = cls_w
                         # Init Math (6) output weights similar to Fact (2)
                         # This means it triggers on similar contexts initially
                         new_w[6] = cls_w[2].clone() + torch.randn_like(cls_w[2]) * 0.02
                         state_dict['intent_classifier.weight'] = new_w
                         
                         if 'intent_classifier.bias' in state_dict:
                             cls_b = state_dict['intent_classifier.bias']
                             new_b = torch.zeros(7, device=cls_b.device)
                             new_b[:6] = cls_b
                             new_b[6] = cls_b[2].clone()
                             state_dict['intent_classifier.bias'] = new_b

                self.model.load_state_dict(state_dict, strict=False)
                print("Model weights loaded.")
            else:
                print(f"Warning: Resume path {self.config.resume_path} not found. Starting fresh.")
        
        print(f"Model parameters: {self.model.count_parameters():,}")
        print("Breakdown:", self.model.get_param_breakdown())
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=(self.device.type == 'cuda')
        )
        
        # Data
        self.train_loader = None
        self.val_loader = None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))
    
    def _cleanup(self):
        """Free RAM and VRAM after training."""
        import gc
        
        print("Cleaning up memory...")
        
        # Delete large objects
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'optimizer'):
            del self.optimizer
        if hasattr(self, 'train_loader'):
            del self.train_loader
        if hasattr(self, 'val_loader'):
            del self.val_loader
        if hasattr(self, 'scaler'):
            del self.scaler
        
        # Force garbage collection
        gc.collect()
        
        # Free CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("Memory cleanup complete.")
    
    def _init_tokenizer(self) -> ByteEncoder:
        """Initialize byte encoder (no tokenization, raw UTF-8 bytes).
        
        ByteEncoder requires no training or saved state.
        Vocab size is always 256 (all possible byte values).
        """
        print("Using raw byte encoding (no tokenization, vocab=256)")
        return ByteEncoder(max_len=self.config.max_seq_len)
    
    def _create_model(self) -> RNK:
        """Create model based on size config."""
        if self.config.model_size == "small":
            config = RNKConfig(
                vocab_size=len(self.tokenizer) or 2048,
                d_model=192,
                d_hidden=384,
                n_hrm_layers=2,
                n_fast_layers=1,
                n_decoder_layers=1
            )
        elif self.config.model_size == "large":
            config = RNKConfig(
                vocab_size=len(self.tokenizer) or 2048,
                d_model=384,
                d_hidden=768,
                n_hrm_layers=4,
                n_fast_layers=2,
                n_decoder_layers=2
            )
        elif self.config.model_size == "xl":
            # ~50M params: balanced for 80k samples
            config = RNKConfig(
                vocab_size=len(self.tokenizer) or 2048,
                d_model=704,          # Tuned for 50M target
                d_hidden=1408,        # 2x d_model
                n_hrm_layers=4,       
                n_fast_layers=3,      
                n_memory_slots=16,    
                n_decoder_layers=3,   
                chunk_size=32
            )
        else:  # base
            config = RNKConfig(
                vocab_size=len(self.tokenizer) or 2048,
                d_model=256,
                d_hidden=512,
                n_hrm_layers=3,
                n_fast_layers=3,  # Increased
                n_memory_slots=16, # Increased
                n_decoder_layers=2
            )
        
        return RNK(config)
    
    def load_data(self):
        """Load and prepare datasets."""
        print("Loading training data...")
        
        # Determine sequence length for curriculum
        if self.config.use_curriculum:
            seq_len = self.config.initial_seq_len
        else:
            seq_len = self.config.max_seq_len
        
        # Load training data
        if self.config.train_data_path.endswith('.jsonl'):
            conversations = load_conversations_from_jsonl(self.config.train_data_path)
            if self.config.max_samples:
                conversations = conversations[:self.config.max_samples]
            train_dataset = ConversationDataset(conversations, self.tokenizer, seq_len)
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True, 
                num_workers=0,
                collate_fn=collate_fn
            )
        else:
            # Auto-split logic
            full_dataset = ConversationDataset(conversations, self.tokenizer, seq_len)
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=collate_fn
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=collate_fn
            )
            print(f"Auto-split data: {train_size} train, {val_size} val")
        
        print(f"Train samples: {len(train_dataset)}")
        if self.val_loader:
            print(f"Val samples: {len(val_dataset)}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        # Scheduled Sampling DISABLED for XL training
        # The slow loop path kills tensor core performance
        # TODO: Re-enable after optimizing decoder loop
        sampling_prob = 0.0  # Always use fast batch path
            
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} (p={sampling_prob:.2f})")
        
        for batch in pbar:
            # Handle new dict batch
            sample_weights = None
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(self.device)
                target_intent = batch['intent'].to(self.device)
                split_idx = batch['split_idx'].to(self.device)
                target_answerable = batch['answerable'].to(self.device)
                if 'weight' in batch and batch['weight'] is not None:
                    sample_weights = batch['weight'].to(self.device)
            else:
                input_ids = batch.to(self.device)
                target_intent = None
                split_idx = None
                target_answerable = None
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            # FP16 ONLY - tensor cores require fp16 on consumer GPUs
            dtype = torch.float16
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda'), dtype=dtype):
                result = self.model(
                    input_ids, 
                    target_ids=input_ids,
                    target_intent=target_intent,
                    target_answerable=target_answerable,
                    split_indices=split_idx,
                    sampling_prob=sampling_prob,
                    sample_weights=sample_weights
                )
                loss = result['loss']
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            n_batches += 1
            
            # Display breakdown
            desc = f"Loss: {loss.item():.4f}"
            if 'stop_loss' in result:
                desc += f" (tok: {result['token_loss']:.3f}, stop: {result['stop_loss']:.3f})"
            if 'intent_loss' in result:
                desc += f" (int: {result['intent_loss']:.3f})"
            if 'answerable_loss' in result:
                desc += f" (ans: {result['answerable_loss']:.3f})"
            if 'latent_loss' in result:
                desc += f" (lat: {result['latent_loss']:.3f})"
            if 'contrastive_loss' in result:
                desc += f" (con: {result['contrastive_loss']:.3f})"
            if 'relevance_loss' in result:
                desc += f" (rel: {result['relevance_loss']:.3f})"
            pbar.set_postfix_str(desc)
        
        avg_loss = total_loss / n_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """Run validation."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        for batch in self.val_loader:
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(self.device)
                target_intent = batch['intent'].to(self.device)
                split_idx = batch['split_idx'].to(self.device)
                target_answerable = batch['answerable'].to(self.device)
                # Keep as raw token index for masked loss computation
            else:
                input_ids = batch.to(self.device)
                target_intent = None
                split_idx = None
                target_answerable = None
                
            with torch.no_grad():
                result = self.model(
                    input_ids, 
                    target_ids=input_ids, 
                    target_intent=target_intent,
                    target_answerable=target_answerable,
                    split_indices=split_idx,
                    sample_weights=None # Validation - no weighting? Or use weighting? Usually no weighting for metric.
                )
            total_loss += result['loss'].item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.model.config.to_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_loss': val_loss
        }
        
        path = os.path.join(self.config.checkpoint_dir, f'rnk_epoch_{epoch+1}.pt')
        torch.save(checkpoint, path)
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.config.checkpoint_dir, 'rnk_best.pt')
            torch.save(checkpoint, best_path)
            print(f"  New best model saved!")
    
    def train(self):
        """Full training loop."""
        self.load_data()
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.config.epochs
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        print(f"\nStarting training for {self.config.epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            scheduler.step()
            
            # Log
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.config.epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Time: {elapsed/60:.1f}m")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch, val_loss)
        
        print(f"\nTraining complete! Total time: {(time.time()-start_time)/60:.1f} minutes")
        
        # Save final model
        self.save_checkpoint(self.config.epochs - 1, val_loss)
        
        # Memory cleanup - free RAM and VRAM
        self._cleanup()
    
    @torch.no_grad()
    def generate_sample(self, prompt: str, max_tokens: int = 64) -> str:
        """Generate text from prompt."""
        self.model.eval()
        
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=self.device)
        
        output_ids, stop_probs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        
        # Stop head truncation
        stop_mask = (stop_probs > 0.5).float()
        if stop_mask.sum() > 0:
             stop_idx = torch.argmax(stop_mask, dim=1)[0].item()
             tokens_to_keep = (stop_idx + 1) * self.model.config.chunk_size
             output_ids = output_ids[:, :tokens_to_keep]
        
        return self.tokenizer.decode(output_ids[0].tolist())


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RNK model")
    parser.add_argument("--train", type=str, default="data/train.txt", help="Training data path")
    parser.add_argument("--val", type=str, default=None, help="Validation data path")
    parser.add_argument("--size", type=str, default="base", choices=["small", "base", "large", "xl"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-samples", type=int, default=100000, help="Limit training samples")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint")
    args = parser.parse_args()
    
    config = TrainConfig(
        train_data_path=args.train,
        val_data_path=args.val,
        model_size=args.size,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device,
        max_samples=args.max_samples,
        resume_path=args.resume
    )
    
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
