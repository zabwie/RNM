# RNK Model Architecture

## Overview
A sample-efficient alternative to Transformers using **latent planning** for text generation.

## Core Pipeline
```
Tokens → Encoder → SSM → HRM → NS Refiner → Decoder → Output
                    ↓
              Intent Embedding (conditioning)
```

## Modules

### Encoder (`encoder.py`)
Converts token IDs to latent chunk representations.

### SSM (`ssm.py`)
- **Fast State**: GRU for local coherence
- **Slow Memory**: Slot attention for long-term context

### HRM (`hrm.py`)
Hierarchical Reasoning Module - abstract pattern recognition.

### Neuro-Symbolic Refiner (`neuro_symbolic.py`)
MLP corrector for logical constraints.

### Decoder (`decoder.py`)
GRU-based autoregressive generation conditioned on refined intent.

### Intent System (`model.py`)
- 6 intents: Greeting (0), Question (1), Fact (2), Refusal (3), Opinion (4), Joke (5)
- Intent embedding added to latent state
- Latent interpolation for stable control

## Parameter Count: ~5.4M
