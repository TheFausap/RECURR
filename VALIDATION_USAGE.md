# Adding Validation to Shakespeare Training

## Quick Fix: Split Data Automatically

Modify `train_shakespeare.py` to use a validation split:

```python
# Add after line 15
if not args.validation_ratio:
    args.validation_ratio = 0.05  # 5% validation

# After loading dataset (around line 76)
if hasattr(dataset, 'train_val_split'):
    dataset.train_val_split(args.validation_ratio)

# Use dataset.validation_data for validation
val_loader = DataLoader(
    dataset.validation_data,
    batch_size=1,  # validation uses batch_size=1
    shuffle=False
)

# Inside training loop, add after epoch complete (line 119):
if epoch % args.validation_freq == 0:
    val_loss = 0
    val_step = 0
    model.eval()
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs.view(-1, dataset.vocab_size), val_targets.view(-1)).item()
            val_step += 1
    model.train()
    print(f"  Val Step {val_step}, Val Loss: {val_loss / val_step:.4f}")
```

## Expected Output

```text
=== Training for 100 epochs ===
  Data length: 1209352
  Batch size: 4
  Block size: 32
  Steps per epoch: 94078
  Est. total steps/epoch: 3010528
  Est. tokens/epoch: 3010528

    Step 5, Loss: 4.82
    Step 10, Loss: 4.56
...
  Epoch 0 / 100, Steps: 94078 / 3010528, Avg Loss: 3.45
  Val Step 1000, Val Loss: 3.89  # <-- NEW
  Epoch 1 / 100, Steps: 94078 / 3010528, Avg Loss: 2.89
  Val Step 2000, Val Loss: 3.12  # <-- NEW
  Epoch 2 / 100, Steps: 94078 / 3010528, Avg Loss: 2.45
  Val Step 3000, Val Loss: 2.98  # <-- NEW
```

## Interpretation Guide

| Pattern | Meaning |
|---------|--------|
| Train ↓, Val ↓ | Learning! Both improving |
| Train ↓, Val ↘ | Overfitting - model memorizing training data |
| Train ↓, Val ↑ | Catastrophic overfit - model completely failed |
| Train ↔, Val ↓ | Normal early-stage learning |
| Train ↑, Val ↓ | Data contamination - model learned test answers |

## Current Plateau Analysis

Your `Step 350-500` loss around **1.48** is the validation concern. With ~3M tokens per epoch, by Step 350 you've only processed ~10 epochs worth of data. If validation loss starts creeping up after Step 400, that's early overfitting.

## Recommended Action

Run with validation every 10 epochs:
```bash
source REC/bin/activate && python train_shakespeare.py \
  --epochs 100 \
  --embed-dim 128 \
  --hidden-dim 256 \
  --data shakespeare_works.txt \
  --block-size 32 \
  --batch-size 4 \
  --validation-freq 10
```

Expected timeline:
- Steps 0-500: Loss drops to ~1.0, validation tracks
- Steps 500-1000: Loss plateaus, validation diverges (early overfitting)
- Steps 1000+: Stopping criterion triggers, model saved

---

**TL;DR:** The plateau at 1.48 is a red flag. Add validation loss every 10 epochs. If validation loss creeps above 2.0 while training stays at 1.48, the model is overfitting — you should probably stop training and try a different block size or architecture.