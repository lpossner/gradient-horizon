# Restructure femtochat into multiple .py files — Design

## Context

The project currently has three thin modules (`data.py`, `model.py`, `utils.py`)
holding reusable pieces, but the actual pipeline — config constants, device
selection, TinyStories loading, tokenizer training, the training loop, and
final-generation — lives inline in `experiment.ipynb`. The goal is to move
that logic into clean, focused `.py` modules while keeping a slimmed notebook
for interactive exploration.

`__archive__/` is a separate, scaled-up next-stage reference per `TRAINING.md`
and is not touched by this restructure.

## Decisions

- **Notebook + script split**: add `train.py` as the end-to-end training
  entry point (`uv run train.py`); keep `experiment.ipynb` for interactive
  exploration only.
- **Config as dataclasses**: split into `GPTConfig` (model architecture) and
  `TrainConfig` (training hyperparameters) instead of the current flat
  constants / dict.
- **Flat file layout**: modules stay at the project root (no `femtochat/`
  package directory), matching the current convention.
- **Checkpointing (new, small capability)**: `train.py` saves a checkpoint
  (model weights + tokenizer state + config + loss history) so the notebook
  has something to load and inspect. This is the one piece of new behavior
  in this restructure — everything else is code movement.

## Module breakdown

### `config.py` (new)
- `GPTConfig` dataclass: `ctx_len, emb_dim, n_heads, n_layers, drop_rate, bias, vocab_size`
- `TrainConfig` dataclass: `lr, weight_decay, batch_size, num_epochs, eval_freq, num_train_stories, num_val_stories, max_sample_tokens`
- `EOT_TOKEN` and `VOCAB_SIZE` (tokenizer vocab size) constants live here too, since they're config, not logic.

### `tokenizer.py` (new)
- `TokenizerState` dataclass: `pat_str, mergeable_ranks, special_tokens, eot_token_id` — a plain-data representation of a trained tokenizer, picklable via `torch.save`.
- `train_tokenizer(corpus: list[str], vocab_size: int) -> TokenizerState`: wraps the existing `rustbpe.Tokenizer().train_from_iterator(...)` + mergeable-ranks/eot-id logic from notebook cell 3.
- `build_encoding(state: TokenizerState) -> tiktoken.Encoding`: reconstructs a `tiktoken.Encoding` from a `TokenizerState` (used both right after training and when restoring from a checkpoint).

### `data.py` (extended)
- `load_tinystories(num_train_stories: int, num_val_stories: int) -> tuple[list[str], list[str]]`: wraps the `datasets.load_dataset("roneneldan/TinyStories")` + slicing from notebook cell 2, returning plain text lists (train, val) rather than HF `Dataset` objects.
- `GPTDataset`, `make_dataloader`: unchanged.

### `model.py`
- Unchanged. Already a clean, self-contained module (`MultiHeadAttention`, `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock`, `GPT`).

### `sampling.py` (renamed from `utils.py`)
- `sample_ids`: unchanged (moved as-is).
- `generate_text(model, encoding, eot_token_id, prompt, max_tokens, ctx_len, device) -> str` (new, thin wrapper): encodes `prompt`, calls `sample_ids`, decodes the result. Replaces the encode/sample/decode block duplicated in notebook cells 7 and 8.

### `checkpoint.py` (new)
- `save_checkpoint(path, model, gpt_config, tokenizer_state, train_losses, val_losses, tokens_seen)`: writes a single `torch.save`-d dict with model `state_dict`, both config dataclasses (as dicts via `dataclasses.asdict`), the tokenizer state, and loss history.
- `load_checkpoint(path, device) -> (model, gpt_config, encoding, eot_token_id, train_losses, val_losses, tokens_seen)`: reconstructs the `GPT` model (loaded weights, eval mode) and the `tiktoken.Encoding` via `tokenizer.build_encoding`.
- Checkpoints are written to a new `checkpoints/` directory, added to `.gitignore`.

### `train.py` (new)
End-to-end script assembled from notebook cells 0–7, unchanged logic:
1. Instantiate `TrainConfig`, seed `torch.manual_seed(123)`.
2. Select device (cuda → mps → cpu).
3. `load_tinystories`, `train_tokenizer`, build `GPTConfig` (vocab_size from encoding).
4. Build `GPT` model, `make_dataloader` for train/val.
5. Training loop: unchanged epoch/batch loop, periodic eval + `generate_text` sample printout via `EVAL_FREQ`.
6. `save_checkpoint(...)` at the end.
Runnable as `uv run train.py` (guarded by `if __name__ == "__main__":`).

### `experiment.ipynb` (slimmed)
Two sections, both exploration-only (no training loop):
1. **Data/tokenizer exploration** (TRAINING.md step 1): `load_tinystories`, `train_tokenizer`, inspect raw text, encode/decode round-trip, build a dataloader and decode a batch back to text (shift-by-one sanity check).
2. **Inspect a trained run**: `load_checkpoint(...)`, sample text via `generate_text`, plot `train_losses`/`val_losses` from the checkpoint.

## Out of scope

- No changes to `model.py`'s architecture or `__archive__/`.
- No LR scheduling, multi-checkpoint management, or experiment tracking — just single-checkpoint save/load to support notebook inspection, per TRAINING.md's current stage ("Squeeze out the juice" / checkpointing is explicitly "not yet relevant" beyond this).
- No changes to `pyproject.toml` dependencies (all needed packages — `datasets`, `rustbpe`, `tiktoken`, `torch`, `tqdm` — are already present).

## Testing

This is a from-scratch GPT training repo with no existing test suite; correctness is verified by running `train.py` for a couple of epochs and confirming the training loop behaves as it did in the notebook (loss decreases, sample output prints), then loading the resulting checkpoint from the notebook and confirming `generate_text` and the loss plots work.
