from dataclasses import dataclass

EOT_TOKEN = "<|EOT|>"
TOKENIZER_VOCAB_SIZE = 8192


@dataclass
class GPTConfig:
    vocab_size: int
    ctx_len: int = 256
    emb_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    drop_rate: float = 0.1
    bias: bool = False


@dataclass
class TrainConfig:
    lr: float = 5e-4
    weight_decay: float = 0.1
    batch_size: int = 32
    num_epochs: int = 100
    eval_freq: int = 100
    num_train_stories: int = 5000
    num_val_stories: int = 1000
    max_sample_tokens: int = 50
