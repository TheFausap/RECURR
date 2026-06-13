"""
Train a Neuralese Recurrence language model on Shakespeare's works.
With thought review: intra‑pass attention over previous neuralese states.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path
from collections import Counter
import time
import json


# ==============================================================================
# Dataset (unchanged)
# ==============================================================================
class WordDataset(Dataset):
    def __init__(self, file_path, block_size, min_freq=1, max_vocab=0):
        self.block_size = block_size
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        words = text.split()
        self.words = words
        word_counts = Counter(words)
        self.vocab = ['<PAD>', '<UNK>']
        filtered = [(w, c) for w, c in word_counts.items() if c >= min_freq]
        filtered.sort(key=lambda x: x[1], reverse=True)
        if max_vocab > 0 and len(filtered) > max_vocab - 2:
            filtered = filtered[:max_vocab - 2]
        self.vocab += [w for w, _ in filtered]
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.data = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ==============================================================================
# Model (with thought review)
# ==============================================================================
class IterativeGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=0.0)

    def forward(self, x, h=None):
        return self.gru(x, h)


class CrossPassMemory(nn.Module):
    def __init__(self, memory_slots, memory_dim, hidden_dim):
        super().__init__()
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.mem_proj = nn.Linear(hidden_dim, memory_dim)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, hidden_dim)
        self.register_buffer('memory', torch.zeros(memory_slots, memory_dim))
        self.register_buffer('ptr', torch.tensor(0, dtype=torch.long))
        self.is_full = False

    def update(self, hidden_states):
        if not self.training:
            return
        new_mem = hidden_states[:, -1, :].mean(dim=0)
        new_mem = self.mem_proj(new_mem)
        idx = self.ptr.item()
        self.memory[idx] = new_mem.detach()
        self.ptr = (idx + 1) % self.memory_slots
        if self.ptr == 0:
            self.is_full = True

    def attend(self, query_hidden):
        if not self.is_full and self.ptr == 0:
            return torch.zeros_like(query_hidden)
        num_slots = self.memory_slots if self.is_full else self.ptr.item()
        mem = self.memory[:num_slots]
        queries = self.query_proj(query_hidden)
        keys = self.key_proj(mem)
        values = self.value_proj(mem)
        attn_weights = torch.matmul(queries, keys.T) / (self.memory_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.matmul(attn_weights, values)
        return context


class NeuraleseRecurrenceLM(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_dim,
                 num_layers,
                 num_passes,
                 memory_slots=0,
                 memory_dim=None,
                 dropout=0.1,
                 use_thought_review=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.gru_cell = IterativeGRU(hidden_dim, hidden_dim, num_layers)
        self.num_passes = num_passes
        self.use_thought_review = use_thought_review

        # Thought review attention (if enabled)
        if self.use_thought_review:
            # 4 heads, hidden_dim must be divisible by num_heads (256/4=64)
            assert hidden_dim % 4 == 0, "hidden_dim must be divisible by 4 for thought review attention"
            self.thought_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                      num_heads=4,
                                                      dropout=dropout,
                                                      batch_first=True)

        # Cross‑pass memory (optional)
        self.use_memory = memory_slots > 0
        if self.use_memory:
            if memory_dim is None:
                memory_dim = hidden_dim
            self.memory = CrossPassMemory(memory_slots, memory_dim, hidden_dim)
            self.memory_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.memory = None

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        emb = self.embedding(x)
        h_in = self.input_proj(emb)
        h_in = self.dropout(h_in)

        h_state = None
        thoughts = []   # stores outputs of previous passes for review

        for pass_idx in range(self.num_passes):
            out, h_state = self.gru_cell(h_in, h_state)

            # --- Thought review: attend over previous pass outputs ---
            if self.use_thought_review and pass_idx > 0:
                # Stack previous thoughts: (b, s, num_prev, h)
                prev = torch.stack(thoughts, dim=2)          # (b, s, k, h)
                b, s, k, h = prev.shape

                # Flatten batch and sequence: treat each token independently
                q = out.reshape(b * s, 1, h)                 # (b*s, 1, h)
                kv = prev.reshape(b * s, k, h)               # (b*s, k, h)

                attn_out, _ = self.thought_attn(q, kv, kv)   # (b*s, 1, h)
                attn_out = attn_out.reshape(b, s, h)

                # Add the reviewed context to the current output
                out = out + self.dropout(attn_out)

            # Store output for future passes
            thoughts.append(out)

            # Prepare input for next pass (residual connection)
            if pass_idx < self.num_passes - 1:
                residual = self.input_proj(emb)
                h_in = self.dropout(out + residual)

        # After all passes, `out` is the final representation.
        if self.use_memory:
            self.memory.update(out)
            queries = out.reshape(-1, out.size(-1))
            mem_ctx = self.memory.attend(queries)
            mem_ctx = mem_ctx.reshape(batch_size, seq_len, -1)
            combined = torch.cat([out, mem_ctx], dim=-1)
            gate = torch.sigmoid(self.memory_gate(combined))
            out = gate * out + (1 - gate) * mem_ctx

        logits = self.output_proj(out)
        return logits


# ==============================================================================
# Training helpers (unchanged except UNK masking in generation)
# ==============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def generate_text(model, start_words, word2idx, idx2word, block_size, device,
                  max_new_tokens=100, temperature=0.8):
    model.eval()
    unk_idx = word2idx.get('<UNK>', None)
    pad_idx = word2idx.get('<PAD>', None)
    mask_indices = [idx for idx in (unk_idx, pad_idx) if idx is not None]

    tokens = [word2idx.get(w, word2idx['<UNK>']) for w in start_words.split()]
    if len(tokens) < block_size:
        tokens = [word2idx['<PAD>']] * (block_size - len(tokens)) + tokens
    else:
        tokens = tokens[-block_size:]

    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    generated = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            for idx in mask_indices:
                probs[:, idx] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(idx2word[next_token])
        x = torch.cat([x[:, 1:], torch.tensor([[next_token]], device=device)], dim=1)
    return ' '.join(generated)


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train Neuralese Recurrence LM with Thought Review')
    parser.add_argument('--data_file', type=str, default='shakespeare_works.txt')
    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_passes', type=int, default=3,
                        help='Number of intra‑pass recurrence iterations')
    parser.add_argument('--memory_slots', type=int, default=0)
    parser.add_argument('--memory_dim', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_end', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine'])
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--min_freq', type=int, default=3)
    parser.add_argument('--max_vocab', type=int, default=8000)
    parser.add_argument('--use_thought_review', action='store_true', default=False,
                        help='Enable intra‑pass attention over previous neuralese states')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Dataset
    dataset = WordDataset(args.data_file, args.block_size,
                          min_freq=args.min_freq, max_vocab=args.max_vocab)
    print(f"  Vocab: {dataset.vocab_size} (min_freq={args.min_freq}, max_vocab={args.max_vocab}), "
          f"Block size: {args.block_size}")
    if args.use_thought_review:
        print("  Thought review: enabled")

    vocab_path = checkpoint_dir / 'vocab.json'
    with open(vocab_path, 'w') as f:
        json.dump(dataset.vocab, f)

    # Train / val split
    data = dataset.data
    split_idx = int(len(data) * 0.9)
    train_tokens = data[:split_idx]
    val_tokens = data[split_idx:]

    class TokenSubset(Dataset):
        def __init__(self, tokens, block_size):
            self.tokens = tokens
            self.block_size = block_size
        def __len__(self):
            return len(self.tokens) - self.block_size
        def __getitem__(self, idx):
            x = self.tokens[idx : idx + self.block_size]
            y = self.tokens[idx + 1 : idx + self.block_size + 1]
            return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    train_dataset = TokenSubset(train_tokens, args.block_size)
    val_dataset = TokenSubset(val_tokens, args.block_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Model
    model = NeuraleseRecurrenceLM(
        vocab_size=dataset.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_passes=args.num_passes,
        memory_slots=args.memory_slots,
        memory_dim=args.memory_dim,
        dropout=args.dropout,
        use_thought_review=args.use_thought_review,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.word2idx['<PAD>'],
                                    label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=args.epochs,
                                                         eta_min=args.lr_end)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | "
              f"time: {elapsed:.1f}s")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': args,
            }, checkpoint_path)

    # Generation
    print("\nGenerating sample text...")
    idx2word = {i: w for w, i in dataset.word2idx.items()}
    sample = generate_text(
        model, "To be or not to be",
        dataset.word2idx, idx2word,
        args.block_size, device,
        max_new_tokens=100
    )
    print("Sample output:")
    print(sample)


if __name__ == '__main__':
    main()
