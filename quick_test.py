"""Quick test script to verify model output."""
import torch
from rnk.model import RNK, RNKConfig
from rnk.tokenizer import RNKTokenizer

# Load
checkpoint = torch.load('checkpoints/rnk_best.pt', map_location='cuda', weights_only=False)
config = RNKConfig(**checkpoint['config'])
model = RNK(config).cuda().eval()
model.load_state_dict(checkpoint['model_state'])
tokenizer = RNKTokenizer()
tokenizer.load('data/tokenizer.json')

# Test prompts
prompts = [
    "What color is the sky?",
    "Hello, how are you?", 
    "What is 2+2?",
    "Tell me a joke"
]

print("=" * 50)
print("RNK Model Test Results")
print("=" * 50)

for prompt in prompts:
    full_prompt = f"User: {prompt}\nModel:"
    input_ids = torch.tensor([tokenizer.encode(full_prompt)], device='cuda')
    
    with torch.no_grad():
        output_ids, _, answerable_prob = model.generate(input_ids, max_new_tokens=64, temperature=0.8, top_k=50)
    
    if output_ids.dim() == 1:
        output_ids = output_ids.unsqueeze(0)
    
    response = tokenizer.decode(output_ids[0].tolist())
    
    print(f"\nPrompt: {prompt}")
    print(f"Answerable: {answerable_prob.item():.2%}")
    print(f"Response: {response}")
    print("-" * 50)

print("\nTest complete.")
