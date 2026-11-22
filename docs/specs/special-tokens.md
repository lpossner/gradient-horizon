# Stage 1: chat and tool special tokens

Date: 2026-11-16
Status: approved

## Intent

Add the four special-token pairs from the roadmap's Stage 1 to the tokenizer, so
`vocab_size` is final before the full pretrain. Both tokenize notebooks are
rewritten cleanly and re-run by the author; the existing corpus is regenerated
rather than migrated.

`vocab_size` 16384 → 16392. Tool *use* stays deferred; the python/output pair
ships reserved and untrained, per the roadmap.

## Approaches considered

**Add the tokens in `train.py`'s `load_tokenizer_state`.** Rejected: that
function's contract is "reconstruct the tokenizer that produced this corpus".
Injecting tokens there makes the code and the pickle disagree about the
vocabulary, hands anything calling `build_encoding` directly a 16384-row vocab
against a 16392-row model (a silent logit-indexing bug, not an error), and
leaves the tokenizer notebook as a second, stale source of truth for token ids.

**Patch the existing pickle with a throwaway cell.** Rejected: it leaves the
committed notebook unable to reproduce the state, which is exactly the
unreproducibility Stage 0 exists to remove. The throwaway cell would have to be
deleted later, and nothing would enforce that it had been.

**Chosen: declare the specials in the tokenize notebook and re-run it.** One
source of truth, no migration code, and the notebook reproduces the state from
scratch. Confirmed `rustbpe` is merge-deterministic across runs, so a re-run
reproduces the same 16383 merges and existing token ids remain valid — the
corpus could in principle be kept, but is regenerated anyway.

## Design

**Assign ids by position after the merges.** The special set becomes an ordered
tuple whose first entry is `eot`, and ids are handed out as
`num_mergeable_ranks + i`:

```python
SPECIAL_TOKENS: tuple[str, ...] = (
    EOT_TOKEN,
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
)
```

The tuple is append-only, and `eot` must stay first: it is 16383 today, and the
corpus encodes every document boundary with that id. An assert pins this.

**`TOKENIZER_VOCAB_SIZE` splits in two.** It is 16383 today — the merge count —
but the vocab becomes 16392. Renamed to `NUM_MERGEABLE_RANKS` (what rustbpe is
asked for), with `VOCAB_SIZE = NUM_MERGEABLE_RANKS + len(SPECIAL_TOKENS)` as the
quantity the uint16 capacity assert checks. The old assert compared the merge
count to dtype capacity and would no longer be testing the real constraint.

**Everything downstream follows for free.** `build_encoding` already threads
`state.special_tokens` through, so `encoding.n_vocab` becomes 16392 and
`GPTConfig(vocab_size=encoding.n_vocab)` picks it up. Neither `train.py` nor the
training notebook changes. `write_split` is untouched — it only ever writes
`eot`.

## Consequences

- **`checkpoint:latest` must be purged before the next Kaggle run.** It is a
  16384-row checkpoint and `load_checkpoint` does a plain `load_state_dict`, so
  the wandb-artifact resume path raises a size mismatch rather than falling
  through to a cold start.
- **Local stays at 16384 for now.** `FineWeb-edu_sample_10BT/` has no parquet,
  so the local 2% corpus cannot be retrained. Its tokens stay valid (eot is
  still 16383 and no other special appears in it), but the local
  `tokenizer_state.pkl` still builds a 16384 model while Kaggle builds 16392.
  Local and Kaggle diverge until that pickle is regenerated.
- **Existing 16384 checkpoints are invalidated**, which is expected: this lands
  before the full pretrain.

## Left to the builder

- Whether the trailing empty cell and stored outputs are cleared on rewrite.
