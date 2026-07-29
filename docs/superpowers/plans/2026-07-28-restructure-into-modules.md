# Restructure femtochat into multiple .py files Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the training pipeline currently embedded in `experiment.ipynb` (config, TinyStories loading, tokenizer training, training loop, generation) into focused `.py` modules, add checkpointing so a trained run can be inspected later, and slim the notebook to pure exploration.

**Architecture:** Seven small modules, each with one responsibility: `config.py` (dataclasses), `tokenizer.py` (BPE train + rebuild), `data.py` (TinyStories loading + windowed dataset, extends existing file), `model.py` (unchanged), `sampling.py` (renamed from `utils.py`, adds a prompt-in/text-out wrapper), `checkpoint.py` (save/load model + tokenizer + losses as one file), and `train.py` (the end-to-end script, new). `experiment.ipynb` is slimmed to two sections that only call into these modules.

**Tech Stack:** Python 3.11+, PyTorch, `datasets` (HF), `rustbpe` + `tiktoken`, `tqdm`, `pytest` (new dev dep), `matplotlib` (new dep, for notebook loss plots).

## Global Constraints

- Flat file layout at project root — no `femtochat/` package directory (spec decision).
- Config represented as two dataclasses: `GPTConfig` (model architecture) and `TrainConfig` (training hyperparameters) — not plain dicts (spec decision).
- `model.py`'s `GPT` class is unchanged and still expects a dict-like config (`config["vocab_size"]`, etc.) — call sites must pass `dataclasses.asdict(gpt_config)`, never the dataclass itself.
- `__archive__/` is not touched by this work.
- Checkpoints are written under `checkpoints/`, which must be added to `.gitignore` — never commit model weights.
- No LR scheduling, multi-checkpoint management, or experiment tracking — single checkpoint save/load only (spec: out of scope).
- No changes to `pyproject.toml` dependencies beyond adding `pytest` (dev) and `matplotlib` (runtime, for notebook plotting) — everything else needed (`datasets`, `rustbpe`, `tiktoken`, `torch`, `tqdm`) is already present.

---

### Task 1: Test infra + `config.py`

**Files:**
- Modify: `pyproject.toml` (add `pytest` to `[dependency-groups].dev`)
- Create: `config.py`
- Test: `test_config.py`

**Interfaces:**
- Produces: `config.EOT_TOKEN: str`, `config.TOKENIZER_VOCAB_SIZE: int`, `config.GPTConfig` dataclass (`vocab_size: int` [no default], `ctx_len: int = 256`, `emb_dim: int = 128`, `n_heads: int = 4`, `n_layers: int = 4`, `drop_rate: float = 0.1`, `bias: bool = False`), `config.TrainConfig` dataclass (`lr: float = 5e-4`, `weight_decay: float = 0.1`, `batch_size: int = 32`, `num_epochs: int = 100`, `eval_freq: int = 100`, `num_train_stories: int = 5000`, `num_val_stories: int = 1000`, `max_sample_tokens: int = 50`).

- [ ] **Step 1: Add pytest dev dependency**

```bash
uv add --dev pytest
```

- [ ] **Step 2: Write the failing test**

Create `test_config.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 4: Write `config.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest test_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock config.py test_config.py
git commit -m "Add config.py with GPTConfig/TrainConfig dataclasses"
```

---

### Task 2: `tokenizer.py`

**Files:**
- Create: `tokenizer.py`
- Test: `test_tokenizer.py`

**Interfaces:**
- Consumes: `config.EOT_TOKEN` (from Task 1)
- Produces: `tokenizer.TokenizerState` dataclass (`pat_str: str`, `mergeable_ranks: dict[bytes, int]`, `special_tokens: dict[str, int]`, `eot_token_id: int`), `tokenizer.train_tokenizer(corpus: list[str], vocab_size: int) -> TokenizerState`, `tokenizer.build_encoding(state: TokenizerState) -> tiktoken.Encoding`.

- [ ] **Step 1: Write the failing test**

Create `test_tokenizer.py`:

```python
from tokenizer import TokenizerState, build_encoding, train_tokenizer

CORPUS = ["hello world", "hello there", "the quick brown fox"] * 5


def test_train_tokenizer_produces_working_encoding():
    state = train_tokenizer(CORPUS, vocab_size=300)
    encoding = build_encoding(state)

    ids = encoding.encode_ordinary("hello world")
    assert encoding.decode(ids) == "hello world"


def test_train_tokenizer_eot_token_id_matches_special_tokens():
    state = train_tokenizer(CORPUS, vocab_size=300)

    assert state.special_tokens == {"<|EOT|>": state.eot_token_id}
    assert state.eot_token_id == len(state.mergeable_ranks)


def test_build_encoding_from_reconstructed_state_matches_original():
    state = train_tokenizer(CORPUS, vocab_size=300)
    encoding = build_encoding(state)

    rebuilt_state = TokenizerState(
        pat_str=state.pat_str,
        mergeable_ranks=state.mergeable_ranks,
        special_tokens=state.special_tokens,
        eot_token_id=state.eot_token_id,
    )
    rebuilt_encoding = build_encoding(rebuilt_state)

    ids = encoding.encode_ordinary("the quick brown fox")
    assert rebuilt_encoding.decode(ids) == "the quick brown fox"
    assert rebuilt_encoding.n_vocab == encoding.n_vocab
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tokenizer'`

- [ ] **Step 3: Write `tokenizer.py`**

```python
from dataclasses import dataclass

import rustbpe
import tiktoken

from config import EOT_TOKEN


@dataclass
class TokenizerState:
    pat_str: str
    mergeable_ranks: dict[bytes, int]
    special_tokens: dict[str, int]
    eot_token_id: int


def train_tokenizer(corpus: list[str], vocab_size: int) -> TokenizerState:
    rustbpe_tokenizer = rustbpe.Tokenizer()
    rustbpe_tokenizer.train_from_iterator(corpus, vocab_size=vocab_size)

    mergeable_ranks = {
        bytes(key): rank for key, rank in rustbpe_tokenizer.get_mergeable_ranks()
    }
    eot_token_id = len(mergeable_ranks)

    return TokenizerState(
        pat_str=rustbpe_tokenizer.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens={EOT_TOKEN: eot_token_id},
        eot_token_id=eot_token_id,
    )


def build_encoding(state: TokenizerState) -> tiktoken.Encoding:
    return tiktoken.Encoding(
        name="dataset_tokenizer",
        pat_str=state.pat_str,
        mergeable_ranks=state.mergeable_ranks,
        special_tokens=state.special_tokens,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test_tokenizer.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add tokenizer.py test_tokenizer.py
git commit -m "Add tokenizer.py for BPE training and encoding reconstruction"
```

---

### Task 3: `data.py` — add `load_tinystories`

**Files:**
- Modify: `data.py` (add function; keep `GPTDataset`/`make_dataloader` as-is)
- Test: `test_data.py`

**Interfaces:**
- Produces: `data.load_tinystories(num_train_stories: int, num_val_stories: int) -> tuple[list[str], list[str]]` (returns `(train_texts, val_texts)`)
- `data.GPTDataset`, `data.make_dataloader` — unchanged signatures, now also covered by a test.

- [ ] **Step 1: Write the failing tests**

Create `test_data.py`:

```python
import data


class _FakeSplit:
    def __init__(self, texts):
        self._texts = texts

    def select(self, indices):
        return _FakeSplit([self._texts[i] for i in indices])

    def __getitem__(self, key):
        assert key == "text"
        return self._texts


def test_load_tinystories_slices_train_and_val(monkeypatch):
    fake_dataset = {
        "train": _FakeSplit([f"train story {i}" for i in range(10)]),
        "validation": _FakeSplit([f"val story {i}" for i in range(10)]),
    }
    monkeypatch.setattr(data, "load_dataset", lambda name: fake_dataset)

    train_texts, val_texts = data.load_tinystories(
        num_train_stories=3, num_val_stories=2
    )

    assert train_texts == ["train story 0", "train story 1", "train story 2"]
    assert val_texts == ["val story 0", "val story 1"]


def test_gpt_dataset_sliding_window():
    ids = list(range(10))
    dataset = data.GPTDataset(ids, max_length=4, stride=2)

    assert len(dataset) == 3
    input_ids, target_ids = dataset[0]
    assert input_ids.tolist() == [0, 1, 2, 3]
    assert target_ids.tolist() == [1, 2, 3, 4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test_data.py -v`
Expected: FAIL — `test_load_tinystories_slices_train_and_val` fails with `AttributeError: <module 'data'> does not have the attribute 'load_dataset'`; `test_gpt_dataset_sliding_window` passes already (existing code) — confirms only the new function is missing.

- [ ] **Step 3: Add `load_tinystories` to `data.py`**

Add this import at the top of `data.py` and the new function above `GPTDataset`:

```python
from datasets import load_dataset


def load_tinystories(
    num_train_stories: int, num_val_stories: int
) -> tuple[list[str], list[str]]:
    dataset = load_dataset("roneneldan/TinyStories")

    train_texts = list(dataset["train"].select(range(num_train_stories))["text"])
    val_texts = list(dataset["validation"].select(range(num_val_stories))["text"])

    return train_texts, val_texts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test_data.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add data.py test_data.py
git commit -m "Add load_tinystories to data.py"
```

---

### Task 4: `sampling.py` (renamed from `utils.py`)

**Files:**
- Create: `sampling.py` (via `git mv utils.py sampling.py`, then edit)
- Delete: `utils.py` (via the rename)
- Test: `test_sampling.py`

**Interfaces:**
- Consumes: `tokenizer.train_tokenizer`, `tokenizer.build_encoding` (Task 2), `config.GPTConfig` (Task 1), `model.GPT` (unchanged)
- Produces: `sampling.sample_ids(model, ids, eot_token_id, max_tokens, ctx_len)` — unchanged behavior, moved as-is. `sampling.generate_text(model, encoding, eot_token_id, prompt: str, max_tokens: int, ctx_len: int, device) -> str`.

- [ ] **Step 1: Rename `utils.py` to `sampling.py`**

```bash
git mv utils.py sampling.py
```

- [ ] **Step 2: Write the failing test**

Create `test_sampling.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest test_sampling.py -v`
Expected: `test_sample_ids_*` PASS (unchanged code, already moved); `test_generate_text_returns_string` FAILS with `AttributeError: module 'sampling' has no attribute 'generate_text'`

- [ ] **Step 4: Add `generate_text` to `sampling.py`**

Append to `sampling.py`:

```python
def generate_text(model, encoding, eot_token_id, prompt, max_tokens, ctx_len, device):
    input_ids = torch.tensor(
        [eot_token_id] + encoding.encode_ordinary(prompt)
    ).unsqueeze(0)

    output_ids = sample_ids(
        model=model,
        ids=input_ids.to(device),
        eot_token_id=eot_token_id,
        max_tokens=max_tokens,
        ctx_len=ctx_len,
    )

    return encoding.decode(output_ids.squeeze(0).tolist())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest test_sampling.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add sampling.py utils.py test_sampling.py
git commit -m "Rename utils.py to sampling.py and add generate_text wrapper"
```

---

### Task 5: `checkpoint.py`

**Files:**
- Create: `checkpoint.py`
- Modify: `.gitignore` (add `checkpoints/`)
- Test: `test_checkpoint.py`

**Interfaces:**
- Consumes: `config.GPTConfig`, `config.TrainConfig` (Task 1), `tokenizer.TokenizerState`, `tokenizer.build_encoding` (Task 2), `model.GPT` (unchanged)
- Produces: `checkpoint.save_checkpoint(path, model, gpt_config: GPTConfig, tokenizer_state: TokenizerState, train_losses: list[float], val_losses: list[float], tokens_seen: list[int]) -> None`, `checkpoint.load_checkpoint(path, device) -> tuple[GPT, GPTConfig, tiktoken.Encoding, int, list[float], list[float], list[int]]` (returns `model, gpt_config, encoding, eot_token_id, train_losses, val_losses, tokens_seen`).

- [ ] **Step 1: Add `checkpoints/` to `.gitignore`**

Append to `.gitignore`:

```
# Model checkpoints
checkpoints/
```

- [ ] **Step 2: Write the failing test**

Create `test_checkpoint.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest test_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'checkpoint'`

- [ ] **Step 4: Write `checkpoint.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest test_checkpoint.py -v`
Expected: PASS (1 test)

- [ ] **Step 6: Commit**

```bash
git add checkpoint.py test_checkpoint.py .gitignore
git commit -m "Add checkpoint.py for saving/loading trained runs"
```

---

### Task 6: `train.py`

**Files:**
- Create: `train.py`

**Interfaces:**
- Consumes: `config.{EOT_TOKEN,TOKENIZER_VOCAB_SIZE,GPTConfig,TrainConfig}` (Task 1), `tokenizer.{train_tokenizer,build_encoding}` (Task 2), `data.{load_tinystories,make_dataloader}` (Task 3), `sampling.generate_text` (Task 4), `checkpoint.save_checkpoint` (Task 5), `model.GPT` (unchanged)
- Produces: `train.get_device() -> torch.device`, `train.main(train_config: TrainConfig | None = None) -> None`, `train.CHECKPOINT_PATH: str`. No automated test (integration script); verified manually in Step 2.

This task has no unit test — it's an orchestration script wiring together already-tested pieces. Verification is a manual smoke run with a tiny config (Step 2).

- [ ] **Step 1: Write `train.py`**

```python
from dataclasses import asdict

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from checkpoint import save_checkpoint
from config import TOKENIZER_VOCAB_SIZE, GPTConfig, TrainConfig
from data import load_tinystories, make_dataloader
from model import GPT
from sampling import generate_text
from tokenizer import build_encoding, train_tokenizer

CHECKPOINT_PATH = "checkpoints/model.pt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(train_config: TrainConfig | None = None) -> None:
    train_config = train_config or TrainConfig()
    torch.manual_seed(123)

    device = get_device()
    print(f"Using {device} device.")

    train_texts, val_texts = load_tinystories(
        train_config.num_train_stories, train_config.num_val_stories
    )

    tokenizer_state = train_tokenizer(
        train_texts + val_texts, vocab_size=TOKENIZER_VOCAB_SIZE
    )
    encoding = build_encoding(tokenizer_state)
    eot_token_id = tokenizer_state.eot_token_id

    train_ids = [eot_token_id]
    for document in train_texts:
        train_ids.extend(encoding.encode_ordinary(document) + [eot_token_id])

    val_ids = [eot_token_id]
    for document in val_texts:
        val_ids.extend(encoding.encode_ordinary(document) + [eot_token_id])

    gpt_config = GPTConfig(vocab_size=encoding.n_vocab)

    model = GPT(asdict(gpt_config))
    model.to(device)

    train_loader = make_dataloader(
        train_ids,
        batch_size=train_config.batch_size,
        max_length=gpt_config.ctx_len,
        stride=gpt_config.ctx_len,
        shuffle=True,
        drop_last=True,
    )
    val_loader = make_dataloader(
        val_ids,
        batch_size=train_config.batch_size,
        max_length=gpt_config.ctx_len,
        stride=gpt_config.ctx_len,
        shuffle=False,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )

    train_losses, val_losses, tokens_seen = [], [], []
    step = 0

    val_prompt = train_texts[0][:25]

    epochs_bar = tqdm(total=train_config.num_epochs, desc="Epochs")
    batches_bar = tqdm(total=len(train_loader), desc="Batches", leave=False)
    val_bar = tqdm(total=len(val_loader), desc="Validation", leave=False)

    for epoch in range(train_config.num_epochs):
        batches_bar.reset()
        epoch_loss_sum = 0.0
        epoch_batch_count = 0

        for input_batch, target_batch in train_loader:
            model.train()
            optimizer.zero_grad()

            logits = model(input_batch.to(device))
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_batch.to(device).reshape(-1),
            )
            loss.backward()
            optimizer.step()

            tokens_seen.append(input_batch.numel())
            step += 1

            batch_loss = loss.item()
            epoch_loss_sum += batch_loss
            epoch_batch_count += 1
            epoch_loss = epoch_loss_sum / epoch_batch_count

            batches_bar.set_postfix(loss=f"{batch_loss:.3f}")
            batches_bar.update(1)
            epochs_bar.set_postfix(loss=f"{epoch_loss:.3f}")

            if step % train_config.eval_freq == 0:
                model.eval()
                train_losses.append(epoch_loss)

                val_loss = 0.0
                val_bar.reset()
                with torch.no_grad():
                    for val_input, val_target in val_loader:
                        val_logits = model(val_input.to(device))
                        batch_val_loss = F.cross_entropy(
                            val_logits.reshape(-1, val_logits.size(-1)),
                            val_target.to(device).reshape(-1),
                        ).item()
                        val_loss += batch_val_loss
                        val_bar.set_postfix(loss=f"{batch_val_loss:.3f}")
                        val_bar.update(1)
                val_loss /= len(val_loader)
                val_losses.append(val_loss)

                output_text = generate_text(
                    model=model,
                    encoding=encoding,
                    eot_token_id=eot_token_id,
                    prompt=val_prompt,
                    max_tokens=train_config.max_sample_tokens,
                    ctx_len=gpt_config.ctx_len,
                    device=device,
                )

                tqdm.write(
                    f"Epoch {epoch + 1} (Step {step:06d}): "
                    f"Train loss {epoch_loss:.3f}, Val loss {val_loss:.3f}"
                )
                tqdm.write(f"Output: {output_text!r}")

        epochs_bar.update(1)

    save_checkpoint(
        CHECKPOINT_PATH,
        model=model,
        gpt_config=gpt_config,
        tokenizer_state=tokenizer_state,
        train_losses=train_losses,
        val_losses=val_losses,
        tokens_seen=tokens_seen,
    )
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual smoke verification**

This downloads/caches TinyStories on first run (already required by the current notebook, so likely already cached locally). Run a tiny config so it finishes in well under a minute of compute (excluding first-time dataset download):

```bash
uv run python -c "
from config import TrainConfig
import train

train.main(TrainConfig(num_train_stories=5, num_val_stories=2, num_epochs=1, eval_freq=1, batch_size=1))
"
```

Expected: prints `Using <device> device.`, progress bars, one `Epoch 1 (Step ...): Train loss ..., Val loss ...` line with an `Output: '...'` line beneath it, then `Saved checkpoint to checkpoints/model.pt`. Confirm the file exists: `ls -la checkpoints/model.pt`.

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "Add train.py as the end-to-end training entry point"
```

---

### Task 7: Slim `experiment.ipynb` + add matplotlib

**Files:**
- Modify: `experiment.ipynb` (replace all cells)
- Modify: `pyproject.toml` (add `matplotlib` dependency)

**Interfaces:**
- Consumes: `config.{TrainConfig,TOKENIZER_VOCAB_SIZE}` (Task 1), `tokenizer.{train_tokenizer,build_encoding}` (Task 2), `data.{load_tinystories,make_dataloader}` (Task 3), `sampling.generate_text` (Task 4), `checkpoint.load_checkpoint` (Task 5), `train.get_device` (Task 6)

No automated test — notebooks aren't unit-testable in this codebase. Verification is running the notebook end-to-end (Step 3), which depends on `checkpoints/model.pt` existing from Task 6's smoke run.

- [ ] **Step 1: Add matplotlib dependency**

```bash
uv add matplotlib
```

- [ ] **Step 2: Replace `experiment.ipynb` contents**

Use the NotebookEdit tool to delete all existing cells and replace them with the following five code cells, in order:

Cell 1 — data exploration:
```python
from config import TrainConfig
from data import load_tinystories

train_config = TrainConfig(num_train_stories=50, num_val_stories=10)

train_texts, val_texts = load_tinystories(
    train_config.num_train_stories, train_config.num_val_stories
)

print(f"Train stories: {len(train_texts)}")
print(f"Val stories:   {len(val_texts)}")
print(train_texts[0][:200])
```

Cell 2 — tokenizer exploration:
```python
from config import TOKENIZER_VOCAB_SIZE
from tokenizer import build_encoding, train_tokenizer

tokenizer_state = train_tokenizer(train_texts + val_texts, vocab_size=TOKENIZER_VOCAB_SIZE)
encoding = build_encoding(tokenizer_state)

sample_text = train_texts[0][:200]
sample_ids = encoding.encode_ordinary(sample_text)

print(f"Vocab size: {encoding.n_vocab}")
print(f"Tokens for {len(sample_text)} chars: {len(sample_ids)}")
print(f"Round-trip matches original: {encoding.decode(sample_ids) == sample_text}")
```

Cell 3 — dataloader / shift-by-one check:
```python
from data import make_dataloader

eot_token_id = tokenizer_state.eot_token_id

train_ids = [eot_token_id]
for document in train_texts:
    train_ids.extend(encoding.encode_ordinary(document) + [eot_token_id])

train_loader = make_dataloader(
    train_ids, batch_size=2, max_length=32, stride=32, shuffle=False, drop_last=True
)

input_batch, target_batch = next(iter(train_loader))
print("Input :", encoding.decode(input_batch[0].tolist()))
print("Target:", encoding.decode(target_batch[0].tolist()))
```

Cell 4 — load a trained checkpoint and sample text:
```python
from checkpoint import load_checkpoint
from sampling import generate_text
from train import get_device

device = get_device()

(
    model,
    gpt_config,
    ckpt_encoding,
    ckpt_eot_token_id,
    train_losses,
    val_losses,
    tokens_seen,
) = load_checkpoint("checkpoints/model.pt", device)

output_text = generate_text(
    model=model,
    encoding=ckpt_encoding,
    eot_token_id=ckpt_eot_token_id,
    prompt="Once upon a time,",
    max_tokens=200,
    ctx_len=gpt_config.ctx_len,
    device=device,
)
print(output_text)
```

Cell 5 — plot loss curves:
```python
import matplotlib.pyplot as plt

plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Evaluation step")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

- [ ] **Step 3: Manual verification**

Requires `checkpoints/model.pt` to exist (from Task 6, Step 2). Run the notebook headlessly:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace experiment.ipynb
```

Expected: exits 0 with no errors. Open `experiment.ipynb` and confirm each cell's output looks sane: story counts printed, `Round-trip matches original: True`, input/target text differing by one token, generated text printed, and a loss plot rendered in the output of the last cell.

- [ ] **Step 4: Commit**

```bash
git add experiment.ipynb pyproject.toml uv.lock
git commit -m "Slim experiment.ipynb to data/tokenizer exploration and checkpoint inspection"
```

---

## Post-plan cleanup

After all tasks are done, confirm the full test suite still passes and no stray files remain:

```bash
uv run pytest -v
git status
```

Expected: all tests pass, and `git status` shows a clean tree (only the plan/spec docs were the pre-existing untracked additions before this work began — everything else should now be committed).
