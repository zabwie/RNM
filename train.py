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
    'joke': 5
}

def classify_intent(text: str, prompt: str = "") -> int:
    """Rigorous regex-based intent classification."""
    text = text.lower().strip()
    prompt = prompt.lower().strip()
    
    # 1. Joke (Highest Priority Override via Prompt)
    if re.search(r'\b(joke|funny|laugh|pun)\b', prompt):
        return INTENT_MAP['joke']
        
    # 2. Refusal (Negative sentiment)
    if re.search(r'\b(no|not|cannot|can\'t|don\'t|stop|sorry|unable)\b', text):
        return INTENT_MAP['refusal']

    # 3. Greeting (Start of sentence/conversation)
    # Strict anchor at start ^
    if re.search(r'^(hello|hi|hey|greetings|good morning|good evening)\b', text):
        return INTENT_MAP['greeting']
    
    # 4. Question (Ends with ? or starts with WH-word)
    if "?" in text or re.search(r'^(what|why|how|when|where|who)\b', text):
        return INTENT_MAP['question']
        
    # 5. Opinion (Subjective markers)
    if re.search(r'\b(i think|i feel|believe|opinion|my view)\b', text):
        return INTENT_MAP['opinion']
        
    # 6. Fact (Default)
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
        
        # Pad sequences
        max_len = max(len(ids) for ids in input_ids)
        
        # Critical: Pad to chunk_size multiple (32)
        # The encoder truncates to chunk boundaries. If we don't pad here, 
        # the response (which starts after a boundary) might be truncated, 
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
            'answerable': answerabilities
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
from rnk.tokenizer import RNKTokenizer


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
    
    # Curriculum
    use_curriculum: bool = True
    initial_seq_len: int = 128
    max_seq_len: int = 256
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1  # epochs
    
    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"


class TextDataset(Dataset):
    """Simple text dataset for RNK training."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: RNKTokenizer,
        max_len: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Tokenize all texts
        self.samples = []
        for text in texts:
            ids = tokenizer.encode(text, max_length=max_len, padding=True)
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
        tokenizer: RNKTokenizer,
        max_len: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.encodings = []
        self.intents = []
        self.split_indices = []
        self.answerabilities = []  # Answerability Gate labels
        
        for item in conversations:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                inp, out = item
                # Format: "User: <inp>\nModel: <out>"
                # This enforces Q->A structure and role separation
                prompt_text = f"User: {inp}\nModel:"
                full_text = f"{prompt_text} {out}"
                
                intent = classify_intent(out, prompt=inp)
                answerable = classify_answerability(inp, out)
                
                # Track prompt end (Prediction Point)
                prompt_enc = tokenizer.encode(prompt_text, padding=False)
                
                # Boundary Padding: Pad prompt to chunk boundary (32)
                # This ensures the Prompt ends exactly at a chunk limit
                # So Intent(Prompt Chunk) -> Predicts(Response Chunk)
                chunk_size = 32 # Default RNK chunk size
                pad_len = (chunk_size - len(prompt_enc) % chunk_size) % chunk_size
                if pad_len > 0:
                    prompt_enc = prompt_enc + [0] * pad_len
                
                split_idx = len(prompt_enc)
                
                # Encode response and combine
                # Note: We encode separately to preserve boundary, then concat
                # Space before output is implicit in previous structure but let's be explicit
                response_enc = tokenizer.encode(f" {out}", padding=False)
                ids = prompt_enc + response_enc
            else:
                full_text = str(item)
                intent = INTENT_MAP['fact']
                answerable = 1.0  # Default answerable
                split_idx = 0
                ids = tokenizer.encode(full_text, max_length=self.max_len, padding=False)
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
    
    def __len__(self) -> int:
        return len(self.encodings)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': self.encodings[idx],
            'intent': self.intents[idx],
            'split_idx': self.split_indices[idx],
            'answerable': self.answerabilities[idx]
        }


def load_texts_from_file(path: str) -> List[str]:
    """Load texts from file (one per line)."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_conversations_from_jsonl(path: str) -> List[Tuple[str, str]]:
    """Load conversations from JSONL ({"input": ..., "output": ...})."""
    conversations = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            conversations.append((data['input'], data['output']))
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
    
    def _init_tokenizer(self) -> RNKTokenizer:
        """Initialize or load tokenizer."""
        tokenizer = RNKTokenizer(vocab_size=2048)
        
        if os.path.exists(self.config.tokenizer_path):
            print(f"Loading tokenizer from {self.config.tokenizer_path}")
            tokenizer.load(self.config.tokenizer_path)
        else:
            print("Training new BPE tokenizer...")
            
            # Get all texts from training data
            if self.config.train_data_path.endswith('.jsonl'):
                conversations = load_conversations_from_jsonl(self.config.train_data_path)
                texts = [f"{inp} {out}" for inp, out in conversations]
            else:
                texts = load_texts_from_file(self.config.train_data_path)
            
            # Limit samples for tokenizer training too
            if self.config.max_samples:
                texts = texts[:self.config.max_samples]
            
            print(f"Training tokenizer on {len(texts)} texts...")
            tokenizer.train(texts, min_frequency=2)
            
            os.makedirs(os.path.dirname(self.config.tokenizer_path) or '.', exist_ok=True)
            tokenizer.save(self.config.tokenizer_path)
            print(f"Saved tokenizer to {self.config.tokenizer_path} (vocab size: {len(tokenizer)})")
        
        return tokenizer
    
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
        
        # Scheduled Sampling: Linear ramp from 0.0 to 0.5 after warmup
        # Conservative schedule to prevent model from collapsing on own errors
        k = 0.5  # Max sampling probability
        warmup = 5 # Epochs of pure teacher forcing
        if epoch < warmup:
            sampling_prob = 0.0
        else:
            progress = min(1.0, (epoch - warmup) / max(1, self.config.epochs - warmup))
            sampling_prob = k * progress
            
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} (p={sampling_prob:.2f})")
        
        for batch in pbar:
            # Handle new dict batch
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
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            # Use BF16 if available, else FP16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda'), dtype=dtype):
                result = self.model(
                    input_ids, 
                    target_ids=input_ids,
                    target_intent=target_intent,
                    target_answerable=target_answerable,
                    split_indices=split_idx,
                    sampling_prob=sampling_prob
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
                    split_indices=split_idx
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
    parser.add_argument("--size", type=str, default="base", choices=["small", "base", "large"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-samples", type=int, default=100000, help="Limit training samples")
    args = parser.parse_args()
    
    config = TrainConfig(
        train_data_path=args.train,
        val_data_path=args.val,
        model_size=args.size,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device,
        max_samples=args.max_samples
    )
    
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
