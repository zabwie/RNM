"""
ByteEncoder: Raw UTF-8 byte encoding for Mamba-based models.

Replaces BPE tokenization with direct byte-level encoding.
This enables Mamba to learn character→word→sentence patterns naturally.
"""


class ByteEncoder:
    """
    Simple byte-level encoder - no vocabulary, no tokenization.
    
    Maps text directly to UTF-8 bytes (0-255).
    """
    
    def __init__(self, max_len: int = 1024):
        self.max_len = max_len
        self.vocab_size = 256  # All possible byte values
        self.pad_id = 0
        self.bos_id = 2  # <BOS> represented as byte 2 (rarely used in text)
        self.eos_id = 3  # <EOS> represented as byte 3
    
    def __len__(self) -> int:
        return self.vocab_size
    
    def encode(self, text: str, max_length: int = None, padding: bool = False) -> list:
        """
        Encode text to raw UTF-8 bytes.
        
        Args:
            text: Input string
            max_length: Maximum length (truncates if exceeded)
            padding: If True, pad to max_length with zeros
            
        Returns:
            List of byte values (0-255)
        """
        if max_length is None:
            max_length = self.max_len
            
        # Convert to UTF-8 bytes
        byte_ids = list(text.encode('utf-8', errors='replace'))
        
        # Truncate if needed
        if len(byte_ids) > max_length:
            byte_ids = byte_ids[:max_length]
        
        # Pad if needed
        if padding and len(byte_ids) < max_length:
            byte_ids = byte_ids + [self.pad_id] * (max_length - len(byte_ids))
        
        return byte_ids
    
    def decode(self, byte_ids: list) -> str:
        """
        Decode byte IDs back to text.
        
        Args:
            byte_ids: List of byte values (0-255)
            
        Returns:
            Decoded string
        """
        # Filter out padding/special tokens
        clean_ids = [b for b in byte_ids if b > 3]  # Skip 0-3 (pad, unk, bos, eos)
        return bytes(clean_ids).decode('utf-8', errors='replace')
    
    def save(self, path: str):
        """Save is a no-op - ByteEncoder has no learned state."""
        pass
    
    def load(self, path: str):
        """Load is a no-op - ByteEncoder has no learned state."""
        pass
    
    def train(self, texts: list, min_frequency: int = 2):
        """Train is a no-op - ByteEncoder doesn't learn from data."""
        pass
