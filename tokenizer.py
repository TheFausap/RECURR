"""Word-level tokenizer for Shakespeare."""
import re
import json
from pathlib import Path


def tokenize_shakespeare(text):
    """Split text into words, preserving punctuation as separate tokens.

    Splits on whitespace, keeping punctuation attached to adjacent words.
    e.g. "Thou art thy foe, to thy sweet self too cruel:" 
         -> ["Thou", "art", "thy", "foe", ",", "to", "thy", "sweet", "self", "too", "cruel", ":"]
    """
    tokens = []
    for match in re.finditer(r"[^\s]+|[\s]+", text):
        w = match.group()
        if re.match(r"^[\w]+$", w):
            tokens.append(w.lower())
        elif re.match(r"^[\.,;:\"\'\!\?]+$", w):
            tokens.append(w)
    return tokens


def build_vocab(text_file, vocab_size=None, min_freq=1):
    """Build vocabulary from a text file.

    Returns (stoi, itos) dictionaries.
    """
    with open(text_file) as f:
        text = f.read()
    tokens = tokenize_shakespeare(text)

    counter = {}
    for tok in tokens:
        counter[tok] = counter.get(tok, 0) + 1

    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    if vocab_size is None:
        vocab_size = len(sorted_items)
    selected = sorted_items[:vocab_size]

    stoi = {tok: idx for idx, (tok, _) in enumerate(selected)}
    itos = {idx: tok for tok, idx in stoi.items()}

    stoi["<PAD>"] = len(stoi)
    itos[len(stoi)] = "<PAD>"
    stoi["<UNK>"] = len(stoi)
    itos[len(stoi)] = "<UNK>"

    return stoi, itos


def encode(text_file, stoi, itos, max_len=256):
    """Encode text file into token IDs."""
    with open(text_file) as f:
        text = f.read()

    encoded = []
    for line in text.split("\n"):
        line_tokens = tokenize_shakespeare(line)
        line_ids = [stoi.get(t, stoi["<UNK>"]) for t in line_tokens]
        if max_len:
            line_ids = line_ids[:max_len]
        encoded.append(line_ids)

    max_len_actual = max(len(seq) for seq in encoded)
    if max_len_actual > 0:
        pad_len = max_len_actual
    else:
        pad_len = 1

    padded = []
    for seq in encoded:
        padded.extend(seq)
        while len(padded) < pad_len:
            padded.append(stoi["<PAD>"])

    return padded


def decode(encoded_ids, stoi, itos):
    """Convert token IDs back to text."""
    return " ".join([itos[i] if i < len(itos) else f"[ID:{i}]" for i in encoded_ids])


# Legacy aliases (for backwards compatibility)
build_vocab_shakespeare = build_vocab
tokenize_text = tokenize_shakespeare
encode_tokens = encode
decode_tokens = decode
