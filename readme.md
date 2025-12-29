# RNK - Recursive Neuro-Knowledge Model

A **6M parameter** language model that achieves coherent conversation through **latent planning** rather than pure autoregression. No self-attention, O(n) complexity.

## Core Idea

Most language models predict the next token directly. RNK takes a different approach:

1. **Chunk** the input into 32-token blocks
2. **Plan** each chunk in a compressed latent space (HRM)
3. **Refine** the plan through symbolic constraints (NS Refiner)
4. **Decode** the refined intent back to tokens

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
│     SSM     │  State Space Model (fast + slow memory)
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
| Encoder | 524K | 32-token chunks → 256-dim latent |
| SSM | 1.98M | Dual memory (fast GRU + slow accumulator) |
| HRM | 857K | Multi-step planning in latent space |
| NS Refiner | 659K | Symbolic constraint layer |
| Decoder | 2.17M | GRU with intent conditioning |

### Auxiliary Heads

| Head | Type | Purpose |
|------|------|---------|
| **Intent** | 5-class | greeting / question / fact / refusal / opinion |
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

- **Parameters**: 6.19M
- **Architecture**: Stable ✓
- **Training**: Working ✓
- **Q→A Binding**: In progress (training on Alpaca dataset)

### Sample Output (pre-Alpaca)
```
Prompt: Hello, how are you?
Response: I am glad to hear that, sharing helps with yourself and understanding...
```
*Generic but grammatical - needs Q→A data to learn instruction-following.*

## Roadmap

- [x] Intent Head (5-class classification)
- [x] Answerability Gate (hallucination prevention)
- [x] Stop Head (knows when to finish)
- [x] Contrastive Loss (prompt-response binding)
- [x] Role Separation (User/Model format)
- [ ] Train on instruction-following data (Alpaca)
- [ ] Evaluate Q→A accuracy
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