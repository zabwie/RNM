"""
RNK Generation Script

Interactive text generation with trained RNK models.
Supports:
- Interactive REPL mode
- Batch generation
- Various sampling strategies
"""

import argparse
import torch
from pathlib import Path

from rnk import RNK
from rnk.model import RNKConfig
from rnk.tokenizer import RNKTokenizer


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load trained model and tokenizer."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate config and model
    config = RNKConfig.from_dict(checkpoint['config'])
    model = RNK(config)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {model.count_parameters():,} parameters")
    
    return model, config


def interactive_generate(
    model: RNK,
    tokenizer: RNKTokenizer,
    device: torch.device,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9
):
    """Interactive generation REPL."""
    print("\n" + "="*60)
    print("RNK Interactive Generation")
    print("="*60)
    print(f"Settings: temp={temperature}, top_k={top_k}, top_p={top_p}")
    print("Type 'quit' to exit, 'settings' to change parameters")
    print("="*60 + "\n")
    
    # Initialize states for streaming
    fast_hidden = None
    slow_memory = None
    
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not prompt:
            continue
        
        if prompt.lower() == 'quit':
            print("Goodbye!")
            break
        
        if prompt.lower() == 'settings':
            print(f"Current: temp={temperature}, top_k={top_k}, top_p={top_p}, max_tokens={max_tokens}")
            try:
                temperature = float(input("  Temperature (0.1-2.0): ") or temperature)
                top_k = int(input("  Top-K (0-100): ") or top_k)
                top_p = float(input("  Top-P (0.0-1.0): ") or top_p)
                max_tokens = int(input("  Max tokens (1-256): ") or max_tokens)
            except ValueError:
                print("  Invalid input, keeping current settings")
            continue
        
        if prompt.lower() == 'reset':
            fast_hidden = None
            slow_memory = None
            print("  [Context reset]")
            continue
        
        # Encode prompt with structure
        full_prompt = f"User: {prompt}\nModel:"
        input_ids = tokenizer.encode(full_prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=device)
        
        # Generate
        with torch.no_grad():
            output_ids, stop_probs = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                fast_hidden=fast_hidden,
                slow_memory=slow_memory
            )
            
        # Check for stop signal (optional in interactive, but good practice)
        stop_mask = (stop_probs > 0.5).float()
        if stop_mask.sum() > 0:
             stop_idx = torch.argmax(stop_mask, dim=1)[0].item()
             tokens_to_keep = (stop_idx + 1) * model.config.chunk_size
             output_ids = output_ids[:, :tokens_to_keep]
        
        # Decode
        response = tokenizer.decode(output_ids[0].tolist())
        print(f"RNK: {response}\n")


def batch_generate(
    model: RNK,
    tokenizer: RNKTokenizer,
    device: torch.device,
    prompts: list,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9
) -> list:
    """Generate responses for multiple prompts."""
    responses = []
    
    for prompt in prompts:
        full_prompt = f"User: {prompt}\nModel:"
        input_ids = tokenizer.encode(full_prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=device)
        
        with torch.no_grad():
            output_ids, stop_probs, answerable_prob = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Answerability Gate Check
        if answerable_prob.item() < 0.5:
            # Model predicts it can't answer confidently
            responses.append("[Model indicates low confidence - may not have sufficient knowledge to answer this question]")
            continue
            
        if output_ids.dim() == 1:
            output_ids = output_ids.unsqueeze(0)
        
        # Check for stop signal
        # stop_probs is (batch, n_chunks)
        # Find first chunk where p(stop) > 0.5
        stop_mask = (stop_probs > 0.5).float()
        
        if stop_mask.dim() == 1:
            stop_mask = stop_mask.unsqueeze(0)
        
        stop_indices = torch.argmax(stop_mask, dim=1)
        
        # If any stop detected (argmax returns 0 if all 0, so check value)
        if stop_mask.sum() > 0:
            # Get the index for the first batch item
            stop_idx = stop_indices[0].item()
            # Calculate token count: (stop_idx + 1) * chunk_size
            # But wait, stop_idx is the index of the chunk that IS the stop chunk (last valid)
            # So we keep that chunk.
            tokens_to_keep = (stop_idx + 1) * model.config.chunk_size
            output_ids = output_ids[:, :tokens_to_keep]

        response = tokenizer.decode(output_ids[0].tolist())
        responses.append(response)
    
    return responses


def main():
    parser = argparse.ArgumentParser(description="Generate text with RNK")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rnk_best.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Load tokenizer
    tokenizer = RNKTokenizer()
    tokenizer.load(args.tokenizer)
    print(f"Tokenizer loaded: {len(tokenizer)} tokens")
    
    if args.prompt:
        # Single prompt mode
        responses = batch_generate(
            model, tokenizer, device,
            [args.prompt],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {responses[0]}")
    else:
        # Interactive mode
        interactive_generate(
            model, tokenizer, device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )


if __name__ == "__main__":
    main()
