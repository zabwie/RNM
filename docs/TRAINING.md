# RNK Training Guide

## Quick Start
```bash
python train.py --train data/pure_synthetic.jsonl --epochs 40 --batch 512
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 15 | Training epochs |
| `--batch` | 512 | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--size` | base | Model size (small/base/large) |

## Scheduled Sampling
Fixes exposure bias by training on model's own predictions:
- **Warmup**: 5 epochs of pure teacher forcing
- **Ramp**: Linear increase to 50% sampling probability
- **⚠️ Warning**: k>0.5 can cause collapse

## Loss Components
- **Token Loss**: Cross-entropy on response tokens
- **Intent Loss**: Classification accuracy
- **Latent Loss**: Next-chunk prediction
- **Contrastive Loss**: Prompt-response alignment
- **Relevance Loss**: Binary relevance prediction

## Inference Tuning
For clean greetings:
```python
temperature=0.3  # Very deterministic
max_new_tokens=12  # Short responses
top_k=20  # Tight sampling
```
