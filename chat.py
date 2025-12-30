"""Interactive chat with RNK model."""
import torch
from rnk.model import RNK, RNKConfig
from rnk.tokenizer import RNKTokenizer

# Load model
print("Loading model (Cross-Attention Decoder)...")
checkpoint = torch.load('checkpoints/rnk_best.pt', map_location='cuda', weights_only=False)
config = RNKConfig(**checkpoint['config'])
model = RNK(config).cuda().eval()
model.load_state_dict(checkpoint['model_state'])

# Load tokenizer
tokenizer = RNKTokenizer()
tokenizer.load('data/tokenizer.json')

print(f"RNK Model: {model.count_parameters():,} params")
print("Natural generation mode (no forced intent)")
print("Type 'quit' to exit\n")

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
        
        # Build context from history (last 3 turns)
        context_turns = history[-3:] if history else []
        context = ""
        for turn in context_turns:
            context += f"User: {turn['user']}\nModel: {turn['model']}\n"
        
        # Current turn
        prompt = f"{context}User: {user_input}\nModel:"
        input_ids = torch.tensor([tokenizer.encode(prompt)], device='cuda')
        
        # Generate WITHOUT forced intent - let model predict naturally
        with torch.no_grad():
            output_ids, _, _ = model.generate(
                input_ids, 
                max_new_tokens=30,
                temperature=0.5,
                top_k=30,
                repetition_penalty=1.2
                # NO target_intent = natural prediction
            )
        
        response = tokenizer.decode(output_ids[0].tolist()).strip()
        print(f"RNK: {response}\n")
        
        # Save to history
        history.append({'user': user_input, 'model': response})
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
