"""
BPE Tokenizer wrapper for RNK.

Uses HuggingFace tokenizers library for efficient BPE.
Supports training on corpus and encoding/decoding.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors


class RNKTokenizer:
    """
    BPE Tokenizer for RNK with special tokens.
    
    Special tokens:
        0: <pad>
        1: <unk>
        2: <bos>
        3: <eos>
    """
    
    SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    
    def __init__(self, vocab_size: int = 2048):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self._trained = False
    
    def train(
        self,
        texts: List[str],
        min_frequency: int = 2,
        show_progress: bool = True
    ):
        """
        Train BPE tokenizer on texts.
        
        Args:
            texts: list of training texts
            min_frequency: minimum token frequency
            show_progress: show training progress
        """
        # Initialize tokenizer with BPE model
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        
        # Pre-tokenization: split on whitespace and punctuation
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        
        # Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.SPECIAL_TOKENS,
            show_progress=show_progress
        )
        
        # Train
        self.tokenizer.train_from_iterator(texts, trainer=trainer)
        
        # Add ByteLevel decoder for proper output
        from tokenizers import decoders
        self.tokenizer.decoder = decoders.ByteLevel()
        
        # Post-processing: add BOS/EOS
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[
                ("<bos>", self.BOS_ID),
                ("<eos>", self.EOS_ID)
            ]
        )
        
        self._trained = True
    
    def train_from_file(
        self,
        file_path: Union[str, Path],
        min_frequency: int = 2
    ):
        """Train tokenizer from a text file (one text per line)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        self.train(texts, min_frequency)
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: input text
            add_special_tokens: add BOS/EOS
            max_length: truncate to this length
            padding: pad to max_length
            
        Returns:
            token IDs
        """
        if not self._trained:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        ids = encoding.ids
        
        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length]
            elif padding and len(ids) < max_length:
                ids = ids + [self.PAD_ID] * (max_length - len(ids))
        
        return ids
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True
    ) -> List[List[int]]:
        """Encode multiple texts."""
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        
        results = []
        for enc in encodings:
            ids = enc.ids
            if max_length is not None:
                if len(ids) > max_length:
                    ids = ids[:max_length]
                elif padding and len(ids) < max_length:
                    ids = ids + [self.PAD_ID] * (max_length - len(ids))
            results.append(ids)
        
        return results
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if not self._trained:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        
        # Filter special tokens if needed
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t >= 4]  # Skip first 4 special tokens
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def decode_batch(self, batch_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode batch of token IDs."""
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]
    
    def save(self, path: Union[str, Path]):
        """Save tokenizer to file."""
        if not self._trained:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        self.tokenizer.save(str(path))
    
    def load(self, path: Union[str, Path]):
        """Load tokenizer from file."""
        self.tokenizer = Tokenizer.from_file(str(path))
        self._trained = True
    
    @property
    def vocab(self) -> dict:
        """Get vocabulary as dict."""
        if not self._trained:
            return {}
        return self.tokenizer.get_vocab()
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        if not self._trained:
            return 0
        return self.tokenizer.get_vocab_size()


def create_tokenizer_from_corpus(
    corpus_path: Union[str, Path],
    save_path: Union[str, Path],
    vocab_size: int = 2048
) -> RNKTokenizer:
    """
    Convenience function to create and save tokenizer.
    
    Args:
        corpus_path: path to text file with training data
        save_path: where to save the tokenizer
        vocab_size: vocabulary size
        
    Returns:
        trained tokenizer
    """
    tokenizer = RNKTokenizer(vocab_size)
    tokenizer.train_from_file(corpus_path)
    tokenizer.save(save_path)
    return tokenizer
