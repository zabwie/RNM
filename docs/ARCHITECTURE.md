# RNK Model Architecture

## Overview

A **5.4M parameter** sample-efficient language model using:
- **Raw byte input** (no tokenization, vocab=256)
- **Mamba SSM** (selective state space, O(n) complexity)
- **Latent planning** for coherent text generation

## Core Pipeline
```
Raw Bytes → Encoder → Mamba SSM → HRM → NS Refiner → Decoder → Output
                        ↓
                  Intent Embedding (conditioning)
```

## Modules

### ByteEncoder (`byte_encoder.py`)
Raw UTF-8 byte encoding. No BPE tokenization.
- Vocab size: 256 (all byte values)
- No training required
- encode() / decode() for text ↔ bytes

### Encoder (`encoder.py`)
Converts byte IDs to latent chunk representations.
- 256-dim embeddings for 256 byte values
- 32-byte chunk pooling
- Only 65K parameters

### Mamba SSM (`mamba.py`)
Selective state space model - replaces old FastState+SlowMemory.
- 4 MambaBlock layers with RMSNorm
- Input-dependent A, B, C matrices (selective gating)
- Parallel scan for O(n) complexity
- Pure PyTorch, no CUDA compilation needed

### HRM (`hrm.py`)
Hierarchical Reasoning Module - abstract pattern recognition.
- 5-step latent planning
- 3 HRM layers

### Neuro-Symbolic Refiner (`neuro_symbolic.py`)
MLP corrector for logical constraints.
- 4 constraint dimensions

### Decoder (`decoder.py`)
GRU-based autoregressive generation conditioned on refined intent.
- Cross-attention to encoder output
- ~2M parameters

## Intent System (`model.py`)
- 7 intents: Greeting, Question, Fact, Refusal, Opinion, Joke, Math
- Intent embedding added to latent state
- Latent interpolation for stable control

## Why Raw Bytes?

Mamba's O(n) scaling enables direct byte input:

| Tokenization | Raw Bytes |
|--------------|-----------|
| Arbitrary word splits | Natural character patterns |
| Numbers chunked weirdly | Clean digit sequence |
| 8K+ vocab, big embedding | 256 vocab, tiny embedding |
| Tokenizer training needed | No training needed |

The model learns character→word→sentence patterns naturally.

## Parameter Count

| Encoder | 65K |
| Mamba SSM | 1.3M |
| HRM | 857K |
| NS Refiner | 659K |
| Decoder | 2.0M |
| **Total** | **5.4M** |

## Capacity-Data Mismatch (Critical Finding)

> **For raw-byte language, optimal capacity scales with dataset entropy, not task complexity.**

### Empirical Results

| Model Size | Training Time | Output Quality |
|------------|---------------|----------------|
| 2.7M (tiny) | 24 mins | "Hi there! How can I help?" ✅ |
| 27M (base) | 95 mins | Garbled byte sequences ❌ |

### Why Smaller Is Better (for limited data)

1. **2.7M forced to compress** → learns abstract patterns (grammar, cadence, intent)
2. **27M had excess capacity** → memorized noisy byte trajectories
3. This is **grokking behavior**: generalization emerges when the model is too small to cheat

### Scaling Guidelines

| Data Size | Optimal Params | Early-Stop Loss |
|-----------|----------------|-----------------|
| 5k-20k | 2-5M | ~0.25-0.35 |
| 50k+ | 10-20M | ~0.15-0.25 |

**Rule**: Scale data before scaling parameters.

## Design Philosophy

> **"Intelligence is the ability to represent ignorance without collapsing."**

This principle manifests in RNK's architecture:

1. **Answerability Head** - The model can represent "I don't know" without forcing a confident wrong answer
2. **Capacity constraints** - Smaller models *don't try to explain everything*; larger models "collapse" by memorizing noise
3. **Latent planning** - Uncertainty is preserved in compressed representation rather than prematurely collapsed into tokens
4. **Grokking** - True generalization emerges when the model is forced to admit limitations through compression

The 2.7M model succeeds because it represents its ignorance gracefully. The 27M model failed because it tried to explain every byte pattern, collapsing into incoherent memorization.
