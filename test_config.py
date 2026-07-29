from dataclasses import asdict

from config import EOT_TOKEN, TOKENIZER_VOCAB_SIZE, GPTConfig, TrainConfig


def test_gpt_config_defaults_with_required_vocab_size():
    cfg = GPTConfig(vocab_size=8269)

    assert asdict(cfg) == {
        "vocab_size": 8269,
        "ctx_len": 256,
        "emb_dim": 128,
        "n_heads": 4,
        "n_layers": 4,
        "drop_rate": 0.1,
        "bias": False,
    }


def test_train_config_defaults():
    cfg = TrainConfig()

    assert asdict(cfg) == {
        "lr": 5e-4,
        "weight_decay": 0.1,
        "batch_size": 32,
        "num_epochs": 100,
        "eval_freq": 100,
        "num_train_stories": 5000,
        "num_val_stories": 1000,
        "max_sample_tokens": 50,
    }


def test_constants():
    assert EOT_TOKEN == "<|EOT|>"
    assert TOKENIZER_VOCAB_SIZE == 8192
