# Word-Level Training for Shakespeare

## Why Character-Level Failed

The 2048-character blocks contain **random noise**, not meaningful text. The model learned to predict the most common character in those chunks — which is often a space or vowel — rather than actual Shakespearean syntax.

## Word-Level Vocabulary

| Metric | Character-Level | Word-Level |
|--------|-----------------|------------|
| Vocab size | 40-67 | 9,489 |
| Token granularity | Individual letters | Words + punctuation |
| Context | 2KB random | Coherent words |
| Learning | Frequency matching | Syntactic patterns |

## Quick Start

### Build the vocabulary (first time only):

```bash
source REC/bin/activate && python -c "
from tokenize import build_vocab
stoi, itos = build_vocab('shakespeare_works.txt')
with open('shakespeare_works_vocab.json', 'w') as f:
    json.dump({'stoi': stoi, 'itos': itos}, f)
print('Vocab built:', len(stoi) - 2, 'words')
"
```

### Train on word-level:

```bash
source REC/bin/activate && python train_shakespeare.py \
  --data shakespeare_works_vocab.txt \
  --epochs 100 \
  --block-size 64 \
  --batch-size 4 \
  --embed-dim 128 \
  --hidden-dim 256
```

## Modified `train_shakespeare.py` (word-level)

<tool_call>