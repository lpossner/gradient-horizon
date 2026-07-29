from dataclasses import asdict

import torch

import sampling
import tokenizer
from config import GPTConfig
from model import GPT

CORPUS = ["hello world", "hello there", "the quick brown fox"] * 5


class _StubModel:
    def __init__(self, next_token_id):
        self.next_token_id = next_token_id

    def __call__(self, ids):
        batch_size, seq_len = ids.shape
        logits = torch.zeros(batch_size, seq_len, 5)
        logits[:, -1, self.next_token_id] = 10.0
        return logits


def test_sample_ids_stops_at_eot_token():
    model = _StubModel(next_token_id=2)
    ids = torch.tensor([[0]])

    output = sampling.sample_ids(model, ids, eot_token_id=2, max_tokens=5, ctx_len=8)

    assert output.tolist() == [[0, 2]]


def test_sample_ids_stops_at_max_tokens():
    model = _StubModel(next_token_id=3)
    ids = torch.tensor([[0]])

    output = sampling.sample_ids(model, ids, eot_token_id=99, max_tokens=3, ctx_len=8)

    assert output.tolist() == [[0, 3, 3, 3]]


def test_generate_text_returns_string():
    state = tokenizer.train_tokenizer(CORPUS, vocab_size=300)
    encoding = tokenizer.build_encoding(state)

    gpt_config = GPTConfig(
        vocab_size=encoding.n_vocab, ctx_len=16, emb_dim=8, n_heads=2, n_layers=1
    )
    model = GPT(asdict(gpt_config))
    model.eval()

    output = sampling.generate_text(
        model=model,
        encoding=encoding,
        eot_token_id=state.eot_token_id,
        prompt="hello",
        max_tokens=5,
        ctx_len=gpt_config.ctx_len,
        device=torch.device("cpu"),
    )

    assert isinstance(output, str)
