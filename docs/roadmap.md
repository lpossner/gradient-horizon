# femtochat: roadmap to a simpler educational nanochat

Date: 2026-11-15
Status: design, pending review

## Intent

femtochat is a from-scratch, educational reimplementation of nanochat. Today it
does pretraining and nothing else: one `train.py` trains a GPT on a FineWeb-edu
sample, and the only quality signal is validation loss. The checkpoint it
produces is a base model — it cannot follow an instruction.

The goal is to carry it through the rest of the pipeline — tokenizer,
pretraining, supervised finetuning, chat inference, evaluation, and
reinforcement learning — so it becomes a small chat model that can be talked to
and measured, and so each stage is something worth understanding and writing
about.

**Why.** The blog post framing: understand how LLMs work under the hood, with
writing in public as the accountability mechanism. femtochat is the vehicle.
The same project continues later into a reasoning model, and after that possibly
vision-language or world models; those are named here so the seams are built in
the right places, but they are not designed now.

**Who it serves.** The author first — this is a learning project, and every
design choice below is subordinate to "can I explain why this is here?" Second,
a reader following the blog.

## Goals

- A chat model you can talk to over a CLI, trained end to end from raw text.
- Each pipeline stage is a separate, individually explainable piece.
- The existing ablation capability is preserved: one flag per ablation, one
  entry in `sweep.py`, one matched control — the structure behind `REPORT.md`.
- Training runs on Kaggle's 2×T4 from one script, with local MPS support for the
  small config only.
- Evaluation that produces numbers a small model can actually move: not just
  loss.

## Non-goals

- Matching nanochat's capability or speed. This is the humbler version.
- Any frontier technique: Muon, FP8, FA3, MoE, multi-token prediction.
- Distributed training beyond the single-node 2-GPU case.
- Tool use as a *capability*, in this iteration. The special tokens ship (see
  Stage 1) but the runtime state machine and sandbox do not.
- A web UI, a serving stack, or an OpenAI-compatible API.
- Writing blog posts. The author writes those.

## Anti-goals

These would make the project a failure even if it technically worked:

- **A chat model that cannot be evaluated.** If "it works" means eyeballing
  samples, the RL and reasoning stages have no foundation.
- **Losing the ablation capability.** If adding stages makes single-variable
  experiments harder to run, the project has traded away its strongest asset.
- **Divergent copies of the training code.** The current state is worse than
  two copies: there are three, at different model sizes (see Constraints).
- **Unverifiable claims.** Every stage's result must be a number, reproducible
  from a command.
- **Architecture churn.** Changing the model to make it more interesting is a
  failure mode, not progress. See Constraints.

## Constraints

**Hardware.** Kaggle 2×T4, 30 h/week quota. T4 is Turing (SM 7.5): no bf16
tensor cores, and no FA3 (needs SM 80+). fp16 with a GradScaler is required, and
SDPA is the only attention path. Local development is an M1 Mac via MPS, which
cannot fit the production config — see Environments.

**Session limits.** A full pretraining run (~42 h at 35 TFLOPS) exceeds a single
Kaggle session. Checkpoint-and-resume must work across a session boundary,
which is what the Kaggle script's wandb-artifact fallback exists for. Currently
untested.

**Data.** Kaggle has the full FineWeb-edu sample-10BT: 14.51 GB compressed,
~10.6B tokens. The local copy is a 2% subset (200M tokens). Full-scale runs are
Kaggle-only. Verify the Kaggle artifact with a decompressed byte count rather
than the gzip ISIZE field — ISIZE is modulo 2³² and only describes the last
member if the file is a multi-member concatenation.

**Model — 84.2M parameters.** 768 dim, 12 heads, 10 layers, 1024 context, 16k
vocab, tied embeddings, head_dim 64. This is GPT-2 small minus the vocab and two
layers, trained at 126 tokens/param against the 10.6B corpus, which is the
over-training ratio a chat model wants. It fits comfortably: ~3.9 GiB of 16 GiB
per T4.

**The model is not changed** except for the tokenizer additions in Stage 1 and
the logit softcap noted below.

### Three divergent configs, and which is canonical

This is the single biggest source of confusion in the current repo, and the
plan below depends on resolving it:

| Where | Config | Params |
|---|---|---|
| Local `train.py`, `full` profile | 384d / 6h / 8L / ctx 512 | 20.9M |
| Committed Kaggle notebook | 512d / 8h / 12L / ctx 1024 | 46.7M |
| **Live Kaggle notebook (canonical)** | **768d / 12h / 10L / ctx 1024** | **84.2M** |

The live notebook is newer than both committed artifacts and is the config this
plan targets. The local `full` profile's own comment says it was shrunk to fit
16 GB of MPS memory.

**Consequence for `REPORT.md`.** Its findings — "clipping does nothing under
Adam", "warmup not clipping prevents the spike", "weight tying is worth
nothing" — were measured on a 20.9M model. They are hypotheses at 84M, not
established results. Any of them that a later stage relies on should be re-run
at the target size before being treated as settled.

**Tokenizer.** The tokenizer training script lives in a *separate Kaggle
notebook*, not in this repo, so the encoding is not reproducible from version
control. This is a real risk: the tokenizer determines `vocab_size`, which
determines model shape. Capturing that notebook into the repo is a Stage 0
prerequisite.

The current encoding has one special token (`eot`). Chat and tool tokens change
`vocab_size`, which changes the shape of `tok_emb` and `out_head`, so they must
be added *before* the full pretrain. Same for context length: `pos_emb` is
`nn.Embedding(ctx_len, emb_dim)`, so it is baked into the weights and cannot be
raised afterwards without retraining.

## Approaches considered

### Structure

**A. Evolve in place.** Add stages beside `train.py`, reusing the model as-is.
Cheapest, but `train.py` is 1274 LOC and ablation flags thread through the
architecture (`GPT(config, args)`, `TransformerBlock(config, args)`); each new
stage inherits that coupling.

**B. Restructure into a minimal core with experiments as overlays.** Cleanest
pedagogy, but discards working code and re-litigates a module split already
tried and reverted in `cfd3385`.

**C. New repo.** Cleanest of all, discards `REPORT.md` and the sweep harness.

**Chosen: A, with one addition.** Keep the current file layout and the ablation
flags exactly as they are — the experiment structure stays untouched, per
explicit instruction. Add new stages as sibling scripts. The structural change
is a single shared training script usable in both environments, replacing the
local/Kaggle fork.

The reason to reject B and C is that they optimize for a tidiness the project
does not need, at the cost of the artifact that makes it valuable. A is safe
because the new stages are genuinely new files rather than reorganizations, so
nothing already working is disturbed.

### Sequencing

**Horizontal.** Full pretrain first (~42 h of quota), then SFT, chat, eval, RL
against the finished model. No rework, but weeks before anything can be talked
to, and SFT forces tokenizer decisions after the quota is spent.

**Vertical.** Whole pipeline thin on a toy model, then repeat at scale. Fast
feedback, but toy numbers are meaningless and a 4-layer model may not reproduce
fp16/depth-dependent bugs.

**Hybrid.** Short pretrain, full pipeline against it, full pretrain last.

**Chosen: hybrid.** Integration failures — loss-mask alignment, special-token
handling, KV cache correctness, resume across sessions — are what cost a week of
quota, and they surface against real weights but not against a toy. Hybrid
spends the quota once, at the end, on a pipeline already proven.

## Environments

The 84M config does not fit the 16 GB M1 (the 20.9M profile already OOM'd it).
So:

| Stage | Kaggle 2×T4 | Local M1 |
|---|---|---|
| 0–1 | yes | yes |
| 2 (short pretrain) | yes | no |
| 3–8 | yes | no |

Local development of stages 3–6 uses the `dry` config (128d/4h/4L/ctx 256) for
correctness, not for results. "Runs in both environments" therefore means the
same script runs in both, not that every stage fits in both.

## The plan

### Stage 0 — One training script, one canonical tokenizer

Three things, in order:

1. **Capture the live Kaggle notebook into git.** It is newer than the committed
   copy and is the source of the 84M config. Until it is committed, every merge
   targets a stale version.
2. **Capture the tokenizer notebook into git.** It lives outside the repo, so
   the encoding is currently unreproducible.
3. **Merge the training scripts into one file.** The copies have complementary
   strengths; neither is a superset:

| | local `train.py` | Kaggle script |
|---|---|---|
| Ablation switches | 15 | none |
| DDP | explicitly not | correct, with `no_sync` |
| fp16 GradScaler | no (bf16 hardcoded) | yes |
| `torch.compile` | no | yes |
| wandb resume fallback | no | yes |
| Device / AMP dtype | hardcoded MPS/bf16 | hardcoded CUDA/fp16 |

The merged script keeps all 15 ablation flags and detects the rest from
hardware: device (`cuda` → `mps` → `cpu`), AMP dtype (bf16 on SM 80+, fp16 on
SM 75 with GradScaler, fp32 on CPU/MPS), `torch.compile` off on MPS. DDP stays
optional and is a no-op at world size 1.

The Kaggle notebook becomes a thin wrapper that writes the repository's script
and runs it, rather than embedding a copy. This is what stops the fork
recurring.

**Resume semantics.** Weights + optimizer + step is the bar. The existing
`repeat_loader(skip_batches=...)` replays a shuffled loader and does not
reproduce the same token stream across a session, so exact stream continuation
is explicitly out of scope — document it as approximate rather than claiming
reproducibility.

**Done when:** the same script runs a short training job on both M1 and Kaggle,
all 15 `sweep.py` entries run on Kaggle, and a run resumes correctly across a
real session boundary.

### Stage 1 — Finalize the tokenizer

Add chat and tool special tokens, and fix context length. Both are baked into
the weights, so both happen before the full pretrain.

Ship all four pairs now, following nanochat's set:

```
<|user_start|>  <|user_end|>
<|assistant_start|>  <|assistant_end|>
<|python_start|>  <|python_end|>
<|output_start|>  <|output_end|>
```

`vocab_size` 16384 → 16392. Tool *use* is deferred to a later iteration, but the
*tokens* ship now: adding them later means retraining, and the cost is 8
embedding rows (+6,144 params).

**The python/output tokens ship reserved and untrained.** Nothing in this
pipeline gives them a training signal: FineWeb contains no code, and pretraining
emits only `eot` — the same situation nanochat is in, where the base model
carries the tool tokens in its vocab that pretraining never trains. They are
inert rows at random init, not a capability. The point of shipping them now is
that a later tool-use iteration needs no checkpoint surgery on a model that cost
a week of quota.

Note what tool use actually requires, since it is not a tokenizer concern: the
model learns only the *format* from SFT data (GSM8K's `<<expr=result>>` rendered
as `<|python_start|>expr<|python_end|><|output_start|>result<|output_end|>`), and
the arithmetic is done *externally* — at inference an executor evaluates the
expression and forces the result in as tokens. The model never computes
anything. So tool use needs both SFT data in that format (Stage 3 keeps GSM8K
for exactly this) and an executor plus sandbox (deferred, not designed here).

The existing BPE merges are kept and the specials appended — merges are
independent of specials, and embeddings are initialized randomly for a fresh
pretrain anyway. So no BPE retraining, and existing token ids stay valid, which
means the pre-tokenized Kaggle corpus remains usable.

Context length: 1024, unchanged. Raising it to 2048 costs ~16% more compute;
decide before Stage 7, not now.

**Done when:** the tokenizer round-trips a conversation to ids and back, and a
test asserts every special token is a single id.

### Stage 2 — Short pretraining run

~500M tokens, ~2 h on 2×T4. A weak but genuine base model at the real
architecture, used to build and debug stages 3–6.

**Done when:** validation loss falls, samples are English, and the checkpoint
resumes correctly from a mid-run save.

### Stage 3 — Supervised finetuning

Conversation data, the loss mask, and the chat format.

The mask convention (nanochat's, worth copying): `1` for tokens the assistant is
trained on, `0` for user turns, system text, specials, and tool output. Masked
positions become `-1` in the targets so cross-entropy ignores them.
`render_conversation` returns ids and mask together; keeping them in lockstep is
where this stage's bugs live.

Data: SmolTalk for general conversation, plus MMLU for multiple choice, plus
GSM8K for tool-call *format*. GSM8K stays in even though the executor is
deferred: training the format is cheap, and without it the python/output tokens
stay at random init and a later tool-use iteration starts cold.

Mixed with `TaskMixture`-style deterministic shuffling, reusing the existing
`datasets` dependency.

**Done when:** SFT loss falls on assistant tokens only, and a held-out
conversation renders with the mask visibly correct.

### Stage 4 — Chat inference

This is where the current code has a real gap. `sample_ids` re-runs a full
forward pass over the whole context for every generated token — O(T²) per token,
O(T³) per response. Invisible at 50-token samples; dominant for multi-turn chat.

Build a KV cache, plus a chat loop and CLI. No tool state machine — deferred.

Keep the naive `sample_ids` as the reference implementation and test the cached
path against it: same weights, same prompt, identical tokens. This is the
pattern nanochat uses (`engine.py`'s naive-vs-Engine equivalence test).

**Done when:** cached and naive generation produce identical token sequences,
and a conversation can be held over the CLI.

### Stage 5 — Evaluation

Metrics a small model can actually move. nanochat's full suite is not all
measurable at 84M:

| Metric | Meaningful at 84M? |
|---|---|
| Bits per byte (val) | Yes — and tokenizer-invariant |
| ARC-Easy (4-way) | Yes — should beat the 25% baseline |
| MMLU (4-way) | Marginally — near chance |
| GSM8K | No — hopeless below ~1B |
| HumanEval | No — same |

**Scoring path: chat-style letter argmax** (nanochat's `chat_eval.py` route —
render an MC conversation, take logits at the answer position, argmax over the
letter ids), not base-model few-shot ICL loss. Few-shot ICL multiple-choice will
likely sit at chance at 84M, which would make the headline metric read as
failure; the chat-style path is attainable after SFT.

**Both models are measured**: bpb on the base checkpoint, task accuracy on the
SFT checkpoint. The harness reports which checkpoint produced each number.

**`token_bytes` must be generated.** `evaluate_bpb` requires a per-token byte
length tensor (0 for specials) that femtochat does not have and nothing
currently produces. Generate it alongside the tokenizer, and regenerate after
Stage 1 appends specials. Use raw token bytes, not a decode round-trip — a
decode round-trip corrupts tokens that are not valid standalone UTF-8.

Report the 4-way baseline next to every accuracy: a raw 27% on a 4-way task
reads as failure without the 25% next to it.

**Done when:** one command prints bpb, ARC-Easy, and MMLU with baselines.

### Stage 6 — Reinforcement learning

GRPO-style, following nanochat's simplifications (no KL term, no PPO ratio, no
trust region — it reduces to REINFORCE with a mean-subtracted advantage).

**Reward: exact match on a multiple-choice letter.** nanochat uses GSM8K, which
is unreachable at 84M — the reward would be 0 on every rollout, so the advantage
`r - mean` would be 0 and nothing would train. A task the model *sometimes*
solves is required. MC letter generation is gradeable from text with no
calculator, reuses the Stage 5 harness, and after SFT should land around 30–40%.

**Guard against a collapsed signal.** If accuracy is ~0% or ~100%, the advantage
collapses again. Measure pre-RL accuracy first and choose task difficulty to
land in the 20–70% band. If MC falls outside it, a templated task
(repeat-the-token, format-following) with a tunable difficulty knob gives direct
control over the pass rate.

**Risk.** RL on a small model with a noisy reward can degrade rather than
improve it. Report the pre-RL baseline on the same task and the same sample
count, so a regression is visible.

**Done when:** mean reward rises above the pre-RL baseline, measured the same
way.

### Stage 7 — Full pretraining run

~10.6B tokens, ~42 h at 35 TFLOPS, across at least two Kaggle sessions. Started
only once stages 3–6 are proven against the short run.

**Done when:** the run completes across a session boundary without losing
progress, and bpb beats the short run's.

### Stage 8 — Re-run downstream stages

SFT, eval, and RL against the good base model. Cheap (~2 h) and the payoff of
the hybrid sequencing.

## Decisions

### Decided by the author

- Simpler educational nanochat, extended through RL and the eval suite.
- Experiment structure stays exactly as it is: ablation flags, `sweep.py`,
  matched controls.
- Hybrid sequencing.
- Tool use in scope, but in a later iteration.
- Model sized for a reasonable chat model; must run on 2×T4 on Kaggle.
- The tokenizer training script lives in a separate Kaggle notebook.
- Later stages (reasoning, then VLM/world models) are named but not designed.
- The author writes the blog posts.

### Calls made here, correctable

- **84M (768/12/10/ctx 1024) is canonical.** It is the live Kaggle notebook's
  config and the only one already run. The committed notebook (46.7M) and local
  `full` (20.9M) are stale.
- **Structure A with a unified script.** New stages as sibling files; one
  training script for both environments; the Kaggle notebook becomes a wrapper.
- **Model unchanged** except tokenizer additions and the logit softcap.
- **Logit softcap added** — cheap insurance under fp16, and nanochat uses it.
- **All four special-token pairs ship now**, tool capability deferred. The
  python/output pair ships *reserved and untrained* — inert rows, matching
  nanochat, so a later tool-use iteration needs no checkpoint surgery.
- **Context stays 1024.**
- **RL reward is MC letter exact-match**, with a synthetic dial if the pass rate
  lands outside 20–70%.
- **Stage 5 uses chat-style letter argmax**, not few-shot ICL.
- **Resume = weights + optimizer + step**, stream continuation explicitly
  approximate.
- **Gradient clipping kept but understood as inert.** `REPORT.md` measured that
  clipping a globally-rescaled gradient does not change Adam's step size. The
  real fp16 protection is the GradScaler.
- **GQA, QK-norm, RMSNorm, Muon, FA3 all skipped** — none pay for themselves at
  this size. FA3 is unavailable on SM 7.5 in any case.
- **GSM8K kept in SFT** for tool-call format, even though the executor is
  deferred.

## Open decisions

None blocking. The remaining choice — context length 1024 vs 2048 — is deferred
to Stage 7 and must be made before that run, since it is baked into `pos_emb`.

## Left to the builder

- How the merged script detects environment and paths, and how the Kaggle
  notebook invokes it.
- Whether Stage 0's device/AMP/DDP/checkpoint logic becomes a shared module the
  new stages import, or each stage re-implements it. The former is preferred —
  otherwise the "divergent copies" anti-goal is relocated rather than retired.
- The SFT data mixture and its proportions.
- KV cache layout, and the test that pins it to the naive path.
- Eval harness structure and how results are recorded.
- Test strategy per stage. `tests/` was deleted in `cfd3385`; restoring tests
  alongside new stages is expected but the shape is open.
- Whether `REPORT.md`, the sweep logs, and `figures/` get committed (currently
  untracked).
- Whether to re-run any of `REPORT.md`'s findings at 84M before relying on them.
