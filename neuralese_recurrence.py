#!/usr/bin/env python3
"""Shakespeare training and generation script using neuralese recurrence architecture.

This module provides a simple recurrent neural network for character-level
text generation, trained on Shakespeare's plays.

Usage:
    # Train the model
    python neuralese_recurrence.py --epochs 1000

    # Generate text
    python neuralese_recurrence.py --mode generate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import json
from pathlib import Path


class ShakespeareDataset(Dataset):
    """Dataset for Shakespeare character-level text."""
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
    args = parser.parse_args()

    workdir = Path(args.model)
    workdir.mkdir(exist_ok=True)
    checkpoint_path = workdir / "checkpoint.pt"
    vocab_path = workdir / "vocab.json"

    print(f"\n=== Loading data from {args.data} ===")
    dataset = ShakespeareDataset(args.data, block_size=args.block_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

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
        for epoch in range(args.epochs):
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, dataset.vocab_size), targets.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / min(len(loader), 10)
            if epoch % 100 == 0:
                print(f"  Epoch {epoch}, Loss: {avg_loss:.4f}")

        print(f"\n=== Saving model to {checkpoint_path} ===")
        checkpoint = {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
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


if __name__ == "__main__":
    main()
