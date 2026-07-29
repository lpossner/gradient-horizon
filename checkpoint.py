from dataclasses import asdict
from pathlib import Path

import torch

from config import GPTConfig
from model import GPT
from tokenizer import TokenizerState, build_encoding


def save_checkpoint(
    path,
    model,
    gpt_config: GPTConfig,
    tokenizer_state: TokenizerState,
    train_losses: list[float],
    val_losses: list[float],
    tokens_seen: list[int],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "gpt_config": asdict(gpt_config),
            "tokenizer_state": asdict(tokenizer_state),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "tokens_seen": tokens_seen,
        },
        path,
    )


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    gpt_config = GPTConfig(**checkpoint["gpt_config"])
    model = GPT(asdict(gpt_config))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer_state = TokenizerState(**checkpoint["tokenizer_state"])
    encoding = build_encoding(tokenizer_state)

    return (
        model,
        gpt_config,
        encoding,
        tokenizer_state.eot_token_id,
        checkpoint["train_losses"],
        checkpoint["val_losses"],
        checkpoint["tokens_seen"],
    )
