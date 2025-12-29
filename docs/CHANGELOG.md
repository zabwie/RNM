# RNK Development Changelog

## 2025-12-29: Intent & Greeting Quality

### Phase 5: Intent Control (Hard Anchor)
**Goal**: Fix mode collapse where model defaulted to greetings regardless of input.

**Changes Made**:
- `rnk/model.py`: Added Intent Classifier, Intent Embedding, and Latent Interpolation
- `train.py`: Implemented rigorous regex-based intent classification
- `train.py`: Added contrastive loss (InfoNCE) for prompt-response alignment

**Key Implementation**:
```python
# Latent Interpolation (Safe Anchor)
alpha = 0.5
next_intent = (1 - alpha) * predicted_plan + alpha * intent_embedding
```

**Result**: Model can now be directed to generate Greetings, Facts, or Jokes via intent ID.

---

### Phase 6: Greeting Quality
**Goal**: Clean, minimal greetings without garbage tokens.

**Problem**: Output was "Hello! I course! see you" (garbage after correct start).

**Solution**: Inference tuning, NOT retraining:
| Parameter | Before | After |
|-----------|--------|-------|
| Temperature | 1.0 | **0.3** |
| max_tokens | 32 | **12** |
| top_k | 50 | **20** |

**Result**: "Hello" → "Hi! Great to hear from you." ✅

---

## Architecture Summary

| Component | Purpose |
|-----------|---------|
| **Encoder** | Token → Latent chunks |
| **SSM** | Fast State + Slow Memory |
| **HRM** | Hierarchical Reasoning |
| **NS Refiner** | Symbolic correction |
| **Decoder** | GRU autoregressive generation |
| **Intent Head** | Response type classification |
| **Latent Predictor** | Next-chunk planning |

---

## Recommended Settings

```python
# For clean greetings
model.generate(
    input_ids,
    max_new_tokens=12,
    temperature=0.3,
    top_k=20,
    target_intent=torch.tensor([0])  # 0=Greeting
)
```

## Known Limitations
- Factual responses ("Sky" → "Blue") require data rebalancing
- Small model capacity (5.4M params) limits multi-domain excellence
