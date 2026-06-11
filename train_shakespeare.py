#!/usr/bin/env python3
"""Shakespeare training and generation script using neuralese recurrence architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import json
from pathlib import Path


class ShakespeareDataset(Dataset):
    """Dataset for character-level text."""
    def __init__(self, file_path, block_size=128):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        data = [self.stoi[c] for c in text]
        self.data = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size
        print(f"  Vocab: {self.vocab_size}, Block size: {self.block_size}")
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


class WordDataset(Dataset):
    """Dataset for word-level text (pre-encoded token IDs)."""
    def __init__(self, encoded, block_size=128):
        self.data = torch.tensor(encoded, dtype=torch.long)
        self.block_size = block_size
        self.vocab_size = len(set(self.data))
        print(f"  Vocab: {self.vocab_size}, Block size: {self.block_size}")
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


class SimpleModel(nn.Module):
    """Simple autoregressive model using LSTM."""
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x, _ = self.lstm(self.embedding(x))
        return self.fc(x)


def main():
    parser = argparse.ArgumentParser(description="Shakespeare training/generation")
    parser.add_argument("--mode", choices=["train", "generate"], default="train")
    parser.add_argument("--model", default="shakespeare")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--data", default="shakespeare_sample.txt")
    parser.add_argument("--validation_ratio", type=float, default=0.05)
    parser.add_argument("--vocab-mode", choices=["char", "word"], default="char")
    args = parser.parse_args()

    workdir = Path(args.model)
    workdir.mkdir(exist_ok=True)
    checkpoint_path = workdir / "checkpoint.pt"
    vocab_path = workdir / "vocab.json"

    print(f"\n=== Loading data from {args.data} ===")
    if args.vocab_mode == "char":
        dataset = ShakespeareDataset(args.data, block_size=args.block_size)
    else:  # word-level
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from tokenizer import build_vocab, encode
        vocab_data = json.load(open('shakespeare_works_vocab.json'))
        stoi = vocab_data['stoi']
        itos = vocab_data['itos']
        encoded = encode(args.data, stoi, itos)
        dataset = WordDataset(encoded, block_size=args.block_size)
    
    if hasattr(dataset, 'train_val_split'):
        dataset.train_val_split(args.validation_ratio)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Use dataset.validation_data for validation
    val_loader = DataLoader(
        dataset.validation_data,
        batch_size=1,  # validation uses batch_size=1
        shuffle=False
    )
    
    print(f" Using device: {device}")

    print(f"\n=== Creating model ===")
    model = SimpleModel(
        vocab_size=dataset.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    if args.mode == "train":
        print(f"\n=== Training for {args.epochs} epochs ===")
        steps_per_epoch = dataset.__len__() // args.batch_size
        total_steps = steps_per_epoch * dataset.block_size
        print(f"  Data length: {dataset.__len__()}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Block size: {dataset.block_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Est. total steps/epoch: {total_steps}")
        print(f"  Est. tokens/epoch: {steps_per_epoch * dataset.block_size}")
        print()

        for epoch in range(args.epochs):
            total_loss = 0
            step_count = 0
            for batch_idx, (inputs, targets) in enumerate(loader):
                step_count += 1
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, dataset.vocab_size), targets.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if step_count % 5 == 0:
                    avg_step_loss = total_loss / step_count
                    print(f"    Step {step_count}, Loss: {avg_step_loss:.4f}")
        
        if epoch % args.validation_freq == 0:
            val_loss = 0
            val_step = 0
            model.eval()
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    val_loss += criterion(val_outputs.view(-1, dataset.vocab_size), val_targets.view(-1)).item()
            print(f"  Epoch {epoch} / {args.epochs}, Steps: {step_count} / {total_steps}, Avg Loss: {total_loss / step_count:.4f}")

        print(f"\n=== Saving model to {checkpoint_path} ===")
        checkpoint = {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        if args.vocab_mode == "char":
            with open(vocab_path, 'w') as f:
                json.dump({
                    "stoi": dataset.stoi,
                    "itos": dataset.itos,
                    "vocab_size": dataset.vocab_size
                }, f, indent=2)
    elif args.mode == "generate":
        print(f"\n=== Loading model from {checkpoint_path} ===")
        if not checkpoint_path.exists():
            print("  No checkpoint found! Train first with --mode train")
            return
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        print(f"\n=== Starting generation ===")
        if args.vocab_mode == "char":
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            start_text = input("Enter starting text (or 'To be or not to be '):").strip()
            if not start_text:
                start_text = "To be or not to be "
            start_ids = [vocab["stoi"].get(c, 0) for c in start_text]
            current = torch.tensor([start_ids], dtype=torch.long)
            max_new = 500
            for i in range(max_new):
                with torch.no_grad():
                    last_id = int(current[-1, -1].item())
                    last_tensor = torch.tensor([[last_id]], dtype=torch.long).to(device)
                    outputs, _ = model.lstm(model.embedding(last_tensor))
                    outputs = model.fc(outputs)
                    next_id = int(torch.argmax(outputs[0, -1], dim=-1).item())
                    current = torch.cat([current, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
                    if vocab["itos"].get(str(next_id), "") == "\n" or next_id >= vocab["vocab_size"]:
                        break
            result = ''.join([vocab["itos"][str(c)] for c in current[0].tolist()])
            print(f"\n=== Generated text ===")
            print(result)
        else:  # word-level generation
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            print("Enter your starting text (words separated by spaces, e.g., 'To be or not to be '):")
            start_text = input().strip()
            start_ids = [vocab["stoi"].get(w, 0) for w in start_text.split()]
            current = torch.tensor([start_ids], dtype=torch.long)
            max_new = 50
            for i in range(max_new):
                with torch.no_grad():
                    last_id = int(current[-1, -1].item())
                    last_tensor = torch.tensor([[last_id]], dtype=torch.long).to(device)
                    outputs, _ = model.lstm(model.embedding(last_tensor))
                    outputs = model.fc(outputs)
                    next_id = int(torch.argmax(outputs[0, -1], dim=-1).item())
                    current = torch.cat([current, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
                    if vocab["itos"].get(next_id, "") in ["<PAD>", "<UNK>"]:
                        break
            result = ' '.join([vocab["itos"][w] if w < len(vocab["itos"]) else f"[ID:{w}]" for w in current[0].tolist()])
            print(f"\n=== Generated text ===")
            print(result)


if __name__ == "__main__":
    main()
