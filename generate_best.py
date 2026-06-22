"""
Generate text from the best checkpoint of a BPE Neuralese Recurrence model.
Scans checkpoints to find the one with the lowest saved validation loss,
or falls back to the one with the smallest epoch after a given threshold.
"""
import torch
import argparse
from pathlib import Path
from train_shakespeare import NeuraleseRecurrenceLM, BPEDataset, generate_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--data_file', type=str, default='shakespeare_works.txt')
    parser.add_argument('--tokenizer_cache', type=str, default='bpe_tokenizer.json')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--start_prompt', type=str, default='To be or not to be')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)

    # Find all epoch checkpoints
    ckpt_files = sorted(ckpt_dir.glob('checkpoint_epoch_*.pt'))
    if not ckpt_files:
        print("No checkpoint files found.")
        return

    best_ckpt = None
    best_val = float('inf')

    # Look for a stored 'best_val_loss' in each checkpoint
    for ckpt_path in ckpt_files:
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            val_loss = checkpoint.get('best_val_loss', None)
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                best_ckpt = ckpt_path
        except Exception as e:
            print(f"Skipping {ckpt_path}: {e}")

    # Fallback: if no best_val_loss field, pick the checkpoint just before the instability
    if best_ckpt is None:
        print("No checkpoint with 'best_val_loss' found. Falling back to epoch 110 (or nearest before 118).")
        for ckpt_path in ckpt_files:
            # Extract epoch number from filename
            try:
                epoch = int(ckpt_path.stem.split('_')[-1])
                if epoch <= 115:   # before the divergence
                    best_ckpt = ckpt_path  # keep the last one that qualifies
            except:
                continue

    if best_ckpt is None:
        print("Could not determine a good checkpoint.")
        return

    print(f"Using checkpoint: {best_ckpt}")
    checkpoint = torch.load(best_ckpt, map_location='cpu', weights_only=False)

    # Extract arguments used for training
    ckpt_args = checkpoint['args']

    # Load dataset (BPE)
    dataset = BPEDataset(
        args.data_file,
        block_size=ckpt_args.block_size,
        bpe_vocab_size=ckpt_args.bpe_vocab_size,
        tokenizer_cache=args.tokenizer_cache
    )

    # Build model with same architecture
    model = NeuraleseRecurrenceLM(
        vocab_size=dataset.vocab_size,
        embed_dim=ckpt_args.embed_dim,
        hidden_dim=ckpt_args.hidden_dim,
        num_layers=ckpt_args.num_layers,
        num_passes=ckpt_args.num_passes,
        memory_slots=ckpt_args.memory_slots,
        memory_dim=ckpt_args.memory_dim,
        dropout=ckpt_args.dropout,
        use_thought_review=ckpt_args.use_thought_review,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Generating from prompt: '{args.start_prompt}'")
    output = generate_text(
        model,
        args.start_prompt,
        dataset,
        ckpt_args.block_size,
        device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    print("\nGenerated text:\n")
    print(output)

if __name__ == '__main__':
    main()
