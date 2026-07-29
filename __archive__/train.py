"""
Minimal training script for M1/CPU.
Run as:
    python -m minichat.train
"""

import os
import time
import math
import torch
import torch.nn.functional as F

from minichat.model import GPT, GPTConfig
from minichat.tokenizer import get_tokenizer
from minichat.dataloader import tokenizing_data_loader
from minichat.common import compute_init, print0, print_banner

# Hyperparameters for a small M1-friendly run
B = 4  # micro-batch size
T = 512  # sequence length
dL = 8  # number of layers (depth)
dH = 8  # number of heads
dD = 256  # embedding dimension (must be divisible by dH)
max_steps = 100
eval_every = 20


def main():
    print_banner()

    # Init
    device_type = "mps" if torch.backends.mps.is_available() else "cpu"
    _, _, _, _, device = compute_init(device_type)
    print0(f"Device: {device}")

    # Tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    # Model
    config = GPTConfig(
        sequence_len=T,
        vocab_size=vocab_size,
        n_layer=dL,
        n_head=dH,
        n_kv_head=dH,
        n_embd=dD,
        window_pattern="L",
    )
    model = GPT(config)
    model.to(device)
    model.init_weights()
    print0(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = model.setup_optimizer(
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
    )

    # Dataloader
    try:
        train_loader = tokenizing_data_loader(
            tokenizer,
            B,
            T,
            "train",
            tokenizer_batch_size=128,
            device=device_type,
        )
        # Test fetching one batch to ensure data exists
        next(train_loader)
    except Exception:
        print0("No data found. Generating dummy data for verification...")
        from minichat.dataset import generate_dummy_dataset

        generate_dummy_dataset(num_shards=1)
        train_loader = tokenizing_data_loader(
            tokenizer,
            B,
            T,
            "train",
            tokenizer_batch_size=128,
            device=device_type,
        )

    # Training loop
    model.train()
    for step in range(max_steps):
        t0 = time.time()

        # Fetch batch
        x, y = next(train_loader)
        x, y = x.to(device), y.to(device)

        # Forward
        loss = model(x, y)

        # Backward
        loss.backward()

        # Clip grad norm (optional, but good for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        tok_per_sec = (B * T) / dt

        print0(
            f"step {step:4d} | loss: {loss.item():.4f} | dt: {dt * 1000:.1f}ms | tok/sec: {tok_per_sec:.1f}"
        )

        if step > 0 and step % eval_every == 0:
            print0("Saving checkpoint removed for minimal version to save space.")


if __name__ == "__main__":
    main()
