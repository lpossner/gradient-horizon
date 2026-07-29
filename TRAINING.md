# A Recipe for Training femtochat

Adapted from Karpathy's [recipe](http://karpathy.github.io/2019/04/25/recipe/) for this repo: a from-scratch GPT (`model.py`) trained on TinyStories (`experiment.ipynb`) on CPU/MPS. `__archive__/` is the scaled-up next stage, not the current baseline.

1. Become one with the data
    - read raw TinyStories text before touching `make_dataloader`
    - sanity-check `tiktoken` gpt2 BPE isn't fragmenting this simpler corpus
    - remember `NUM_TRAIN_STORIES=50` / `NUM_VAL_STORIES=10` is a toy split, not a real one
    - know `stride` vs `ctx_len` in `make_dataloader` (currently non-overlapping)

2. Set up the end-to-end skeleton + dumb baselines
    - fix random seed (`torch.manual_seed(123)` already there)
    - simplify (`BATCH_SIZE=1`, 50 stories — don't add anything until this loop is verified)
    - use `.3f` on losses, not fewer digits
    - verify loss at init ≈ `ln(vocab_size)` ≈ 10.8
    - init well (no special init beyond PyTorch defaults yet in `GPT.__init__` — check first if init loss is off)
    - human baseline: read stories, guess next words vs `generate_text` output
    - input-independent baseline: zero out `tok_emb` input, confirm loss plateaus higher
    - overfit one batch (shrink to one batch, drop dropout, loss → ~0; catches shift-by-one bugs in `GPTDataset`/mask)
    - verify decreasing training loss within first `EVAL_FREQ=10` steps
    - visualize just before the net: decode a batch back to text, check the shift-by-one
    - visualize prediction dynamics: fixed prompt (`"Hello, I am"`) every `EVAL_FREQ` steps, keep it fixed across runs
    - chart dependencies via backprop: confirm causal mask in `MultiHeadAttention` blocks future tokens
    - generalize a special case: try `n_layers=1, n_heads=1` by hand before trusting bigger configs

3. Overfit
    - pick the model: `GPT` is already minimal, matches simple data — grow one knob at a time from `GPT_CONFIG_124M`
    - Adam is safe: current `AdamW(lr=5e-4, weight_decay=0.1)` is fine; don't adopt `__archive__`'s per-group-LR optimizer until this baseline works
    - complexify one degree of freedom at a time (e.g. layers *or* data size, not both — only 50 stories)
    - don't trust LR decay defaults: there's no schedule yet (flat LR); add one deliberately if extending training

4. Regularize
    - get more data: raise `NUM_TRAIN_STORIES` first — cheapest lever, TinyStories has ~2M stories
    - data augmentation: overlapping windows via smaller `stride`
    - smaller model/input dim: shrink `emb_dim`/`n_layers` before adding dropout
    - decrease batch size: already at the floor (`BATCH_SIZE=1`)
    - dropout / weight decay: already wired in (`drop_rate`, `WEIGHT_DECAY`) — tune before anything fancier
    - early stopping: watch `val_losses` trend, not single noisy readings (only 10 val stories)

5. Tune
    - random search over `LR`, `WEIGHT_DECAY`, `drop_rate`, `n_layers`, `n_heads`, `emb_dim` — runs are cheap here
    - lower `EVAL_FREQ` while tuning for tighter feedback

6. Squeeze out the juice
    - not yet relevant — no checkpointing/ensembling in this repo; revisit when moving toward the `__archive__` pipeline
