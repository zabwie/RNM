"""Quick test of RNK model."""
import torch
from rnk import RNK
from rnk.model import create_rnk_base

# Create model
model = create_rnk_base()
print('Model created')

# Random input
batch_size = 2
seq_len = 64
input_ids = torch.randint(0, 2048, (batch_size, seq_len))
print(f'Input shape: {input_ids.shape}')

# Forward pass
result = model(input_ids, target_ids=input_ids)
print(f'Logits shape: {result["logits"].shape}')
print(f'Loss: {result["loss"].item():.4f}')

# Generation test
print('Testing generation...')
output = model.generate(input_ids[:1], max_new_tokens=16)
print(f'Generated shape: {output.shape}')

print('\nAll tests passed!')
