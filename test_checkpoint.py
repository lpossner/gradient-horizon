from dataclasses import asdict

import torch

from checkpoint import load_checkpoint, save_checkpoint
from config import GPTConfig
from model import GPT
from tokenizer import train_tokenizer

CORPUS = ["hello world", "hello there", "the quick brown fox"] * 5


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    tokenizer_state = train_tokenizer(CORPUS, vocab_size=300)
    gpt_config = GPTConfig(
        vocab_size=tokenizer_state.eot_token_id + 1,
        ctx_len=16,
        emb_dim=8,
        n_heads=2,
        n_layers=1,
    )
    model = GPT(asdict(gpt_config))

    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model=model,
        gpt_config=gpt_config,
        tokenizer_state=tokenizer_state,
        train_losses=[3.1, 2.5],
        val_losses=[3.4, 2.8],
        tokens_seen=[128, 256],
    )

    (
        loaded_model,
        loaded_gpt_config,
        encoding,
        eot_token_id,
        train_losses,
        val_losses,
        tokens_seen,
    ) = load_checkpoint(path, device=torch.device("cpu"))

    assert loaded_gpt_config == gpt_config
    assert eot_token_id == tokenizer_state.eot_token_id
    assert encoding.n_vocab == gpt_config.vocab_size
    assert train_losses == [3.1, 2.5]
    assert val_losses == [3.4, 2.8]
    assert tokens_seen == [128, 256]

    for (name, original), (_, loaded) in zip(
        model.state_dict().items(), loaded_model.state_dict().items()
    ):
        assert torch.equal(original, loaded), f"mismatch in {name}"
