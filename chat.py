"""Interactive chat with RNK model (raw byte input with inference optimizations)."""
import torch
import re
from rnk.model import RNK, RNKConfig
from rnk.byte_encoder import ByteEncoder

# Load model
print("Loading model (Raw Byte Input)...")
checkpoint = torch.load('checkpoints/rnk_epoch_500.pt', map_location='cuda', weights_only=False)
config = RNKConfig(**checkpoint['config'])
model = RNK(config).cuda().eval()
model.load_state_dict(checkpoint['model_state'])

# Use ByteEncoder (no tokenization)
encoder = ByteEncoder(max_len=1024)

print(f"RNK Model: {model.count_parameters():,} params")
print("Raw byte input with inference optimizations")
print("Type 'quit' to exit\n")


def clean_output(text: str) -> str:
    """Post-process byte output for readability."""
    # Remove non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c in '\n ')
    # Collapse repeated characters (4+ to 2)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    # Collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def generate_with_chunks(prompt: str, num_chunks: int = 3, chunk_size: int = 20) -> str:
    """Generate in chunks, recalibrating context each time."""
    full_text = prompt
    
    for _ in range(num_chunks):
        ids = torch.tensor([encoder.encode(full_text)], device='cuda')
        with torch.no_grad():
            out, _, _ = model.generate(
                ids,
                max_new_tokens=chunk_size,
                temperature=0.5,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.2
            )
        decoded = encoder.decode(out[0].tolist())
        new_part = decoded[len(full_text):]
        if not new_part:
            break
        full_text += new_part
    
    return full_text


# Conversation history for context
history = []

while True:
    try:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not user_input:
            continue
        
        # Build context from history (last 2 turns)
        # NOTE: Disabled - model only supports single-turn prompts (trained on Q&A pairs)
        # context_turns = history[-2:] if history else []
        # context = ""
        # for turn in context_turns:
        #     context += f"User: {turn['user']}\nModel: {turn['model']}\n"
        context = ""  # No context - single-turn mode
        
        # Prompt with starter phrase (Trick 3: anchor the response)
        prompt = f"{context}User: {user_input}\nModel: "
        
        # Generator with optimal coherence parameters (found via sweep)
        # - Temp=0.3: Low temp prevents exploration of noise bytes
        # - Top-k=20 / Top-p=0.8: Strict filtering for clean byte paths
        # - Rep=1.2: Mild repetition penalty
        input_ids = torch.tensor([encoder.encode(prompt)], device='cuda')
        with torch.no_grad():
            out, _, _ = model.generate(
                input_ids,
                max_new_tokens=80, 
                temperature=0.3,
                top_k=20,
                top_p=0.8,
                repetition_penalty=1.2
            )
        
        # Decode and clean
        raw_response = encoder.decode(out[0].tolist())
        full_response = raw_response
        
        # Extract just the model's response
        response = full_response[len(prompt):]
        response = clean_output(response)
        
        # Debug: show raw vs cleaned
        print(f"DEBUG RAW: {repr(response[:80])}")
        
        print(f"RNK: {response}\n")
        
        # Only save non-empty responses to history (prevents corruption)
        if response.strip():
            history.append({'user': user_input, 'model': response})
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
