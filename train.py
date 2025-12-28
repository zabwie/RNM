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
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

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
    batch_size: int = 16
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
        
        self.samples = []
        for inp, out in conversations:
            # Encode input and output together
            combined = f"{inp} {out}"
            ids = tokenizer.encode(combined, max_length=max_len, padding=True)
            if len(ids) >= 16:
                self.samples.append(ids)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.samples[idx], dtype=torch.long)


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
            weight_decay=config.weight_decay
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
                n_fast_layers=2,
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
        else:
            texts = load_texts_from_file(self.config.train_data_path)
            if self.config.max_samples:
                texts = texts[:self.config.max_samples]
            train_dataset = TextDataset(texts, self.tokenizer, seq_len)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Load validation data if available
        if self.config.val_data_path and os.path.exists(self.config.val_data_path):
            if self.config.val_data_path.endswith('.jsonl'):
                val_convs = load_conversations_from_jsonl(self.config.val_data_path)
                val_dataset = ConversationDataset(val_convs, self.tokenizer, seq_len)
            else:
                val_texts = load_texts_from_file(self.config.val_data_path)
                val_dataset = TextDataset(val_texts, self.tokenizer, seq_len)
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
        else:
            # Auto-split if no validation file provided
            val_size = int(len(train_dataset) * 0.1)  # 10% split
            train_size = len(train_dataset) - val_size
            
            # Use fixed generator for reproducibility
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size], generator=generator
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
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
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                result = self.model(batch, target_ids=batch)
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
            batch = batch.to(self.device)
            result = self.model(batch, target_ids=batch)
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
    parser.add_argument("--batch", type=int, default=16)
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
