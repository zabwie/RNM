# RNK Development Changelog

## 2025-12-29: Scale to 100M Parameters

### Phase 8: Semantic Consistency & Grounding (Current)
- **Decoder Upgrade**: Replaced standard GRU with **Cross-Attention Decoder** (Option 3).
  - Feature: Attends to encoder output at initialization to inject context.
  - Optimization: Uses single cross-attention op + pure GRU, enabling **13x speedup** (0.9 it/s -> 12.6 it/s).
  - Result: Token loss dropped to 0.000 (perfect), generating grounded responses ("2+2 equals 4").
- **Tokenizer**: Increased vocab size from 2048 to **8192** to improve BPE subword merges and fix math token corruption.
- **Intent Classifier**: Disabled intent conditioning during generation ("dead weight") to prevent mode collapse and generic loops.

## Phase 7: XL Model
**Goal**: Scale from 5.4M to 100M parameters.

**Changes Made**:
- `train.py`: Added 'xl' model size config

**XL Config**:
```python
d_model=960      # 4x base
d_hidden=1920    # 4x base
n_hrm_layers=6   # 2x base
n_fast_layers=4
n_memory_slots=32
n_decoder_layers=4
```

**Result**: **100,896,426 params** ✅

**Usage**:
```bash
python train.py --train data/pure_synthetic.jsonl --size xl --epochs 40
```

---

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
