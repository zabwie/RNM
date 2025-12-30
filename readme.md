# RNK - Recursive Neuro-Knowledge Model

An **11.5M parameter** language model that achieves coherent conversation through **latent planning** and **Mamba state-space modeling**. No self-attention, O(n) complexity.

## Core Idea

Most language models predict the next token directly. RNK takes a different approach:

1. **Chunk** the input into 32-token blocks
2. **Process** through Mamba SSM (selective state evolution)
3. **Plan** each chunk in a compressed latent space (HRM)
4. **Refine** the plan through symbolic constraints (NS Refiner)
5. **Decode** the refined intent back to tokens

This means the model "thinks before it speaks" - each response is shaped by high-level intent, not just token-by-token prediction.

## Architecture

```
Input Tokens
     │
     ▼
┌─────────────┐
│   Encoder   │  Chunks tokens → latent vectors (32 tok/chunk)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Mamba SSM  │  Selective State Space (unified fast+slow dynamics)
│  (4 layers) │  Input-dependent gating, O(n) complexity
└──────┬──────┘
       │
  ┌────┴────┐
  │         │
  ▼         ▼
┌──────┐  ┌──────────┐
│Intent│  │   HRM    │  Hierarchical Reasoning (5-step planning)
│ Head │  └────┬─────┘
└──┬───┘       │
   │           ▼
   │    ┌────────────┐
   │    │ NS Refiner │  Neuro-Symbolic refinement
   │    └─────┬──────┘
   │          │
   │    ┌─────┴─────┐
   ▼    ▼           ▼
┌──────────┐  ┌──────┐  ┌─────────┐
│Answerable│  │ Stop │  │ Decoder │
│   Head   │  │ Head │  │  (GRU)  │
└──────────┘  └──────┘  └────┬────┘
                             │
                             ▼
                       Output Tokens
```

### Components

| Module | Size | Purpose |
|--------|------|---------|
| Encoder | 2.1M | 32-token chunks → 256-dim latent |
| **Mamba SSM** | 1.3M | Selective state space (replaces old FastState+SlowMemory) |
| HRM | 857K | Multi-step planning in latent space |
| NS Refiner | 659K | Symbolic constraint layer |
| Decoder | 6.0M | Cross-attention GRU with intent conditioning |

### Mamba SSM Details

The Mamba SSM is a **pure PyTorch** implementation of selective state spaces:

- **4 layers** of MambaBlocks with RMSNorm and residual connections
- **Selective gating**: Input-dependent A, B, C matrices (not static like S4)
- **Parallel scan**: O(n) complexity for efficient long sequences
- **No attention**: Linear scaling, constant memory per token

This replaces the old FastState (GRU) + SlowMemory (slot attention) which had coordination failures.

### Auxiliary Heads

| Head | Type | Purpose |
|------|------|---------|
| **Intent** | 7-class | greeting / question / fact / refusal / opinion / identity / math |
| **Answerable** | Binary | "Can I answer this?" gate (prevents hallucination) |
| **Stop** | Binary | Per-chunk termination signal |

## Training

### Loss Function
```
Total = Token + 0.5×Stop + 0.2×Intent + 0.3×Answerable + 0.1×Contrastive
```

### Key Training Innovations

1. **Causal Intent Prediction** - Intent is predicted from prompt-only state, preventing data leakage
2. **Role Separation** - Strict `User:` / `Model:` formatting for instruction-following
3. **Masked Response Loss** - Loss computed only on response tokens, not prompt
4. **Boundary Padding** - Prompts padded to chunk boundaries for clean separation

### Usage

```bash
# Train
python train.py --train data/alpaca.jsonl --epochs 15

# Generate
python generate.py --prompt "What is the capital of France?"
```

## Current Status

- **Parameters**: 11.5M
- **Architecture**: Mamba SSM ✓ (replaced FastState+SlowMemory)
- **Training**: 500-epoch grokking run in progress
- **Goal**: Coherent conversation through extended training

### Sample Output (target)
```
Prompt: Hello, how are you?
Response: Hi! I'm doing well, thanks for asking. How can I help you today?
```
*Training with grokking strategy to achieve this.*

## Roadmap

- [x] Intent Head (7-class classification)
- [x] Answerability Gate (hallucination prevention)
- [x] Stop Head (knows when to finish)
- [x] Contrastive Loss (prompt-response binding)
- [x] Role Separation (User/Model format)
- [x] **Mamba SSM** (replaced old FastState+SlowMemory)
- [/] Grokking training (500 epochs on clean data)
- [ ] Coherent conversation verification
- [ ] Scale to 20-30M params (if architecture proves out)

## Design Philosophy

> "O(n) complexity, no attention, but with the coherence of planning."

RNK bets on **latent reasoning** over **pattern matching**. Instead of learning token co-occurrences, it learns:
- What kind of response is expected (Intent)
- Whether it can answer confidently (Answerability)
- When to stop talking (Stop)
- How to plan multi-step responses (HRM)

This is closer to how humans structure responses: think first, then speak.

## License

Do whatever you want with it. If it works, tell me. If it doesn't, also tell me. Just don't around claiming it was your idea. Using/modifying this code for your own projects is allowed, but please credit me as the original author.