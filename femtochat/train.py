import argparse
import contextlib
import math
import os
import pickle
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rapidgzip
import tiktoken
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# This script only ever runs locally, on the M1 (mps).
DEVICE_TYPE: str = "mps"

# Reads WANDB_API_KEY / WANDB_PROJECT / HF_TOKEN from a local .env file in the
# project root -- this script never runs on Kaggle.
load_dotenv(override=True)


## PATHS
INPUT_DIR: str = str(Path.cwd() / "FineWeb-edu_sample_10BT")
OUTPUT_DIR: str = str(Path.cwd() / "output")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
TOKENIZER_STATE_PATH: str = f"{INPUT_DIR}/tokenizer_state.pkl"
TRAIN_TOKENS_GZ_PATH: str = f"{INPUT_DIR}/train.bin.gz"
TRAIN_TOKENS_INDEX_PATH: str = f"{OUTPUT_DIR}/train.rgidx"
VAL_TOKENS_GZ_PATH: str = f"{INPUT_DIR}/val.bin.gz"
VAL_TOKENS_INDEX_PATH: str = f"{OUTPUT_DIR}/val.rgidx"
TOKENS_DTYPE: type[np.uint16] = np.uint16

## SAMPLING (not swept -- fixed across all configs for comparability)
MAX_SAMPLE_TOKENS: int = 50
TEMPERATURE: float = 0.8
TOP_K: int | None = 40
SEED: int = 123
VAL_PROMPT: str = "The"

# bf16 autocast on mps.
USE_AMP: bool = True
AMP_DTYPE: torch.dtype = torch.bfloat16

BETAS: tuple[float, float] = (0.9, 0.95)
WEIGHT_DECAY: float = 0.1

# Per-profile model size and schedule length. NOTE: ablation switches that
# change parameter/buffer shapes (--n-kv-heads, --position-encoding=rope,
# --ablate-no-weight-tying, --ablate-no-norm) make checkpoints from different
# configs mutually incompatible with the strict load_state_dict below -- each
# ablation run is meant to start from scratch, not resume across configs.
PROFILE_DEFAULTS: dict[str, dict[str, int]] = {
    "full": {
        # Shrunk from the original 1024/512/8/12/batch=24 notebook numbers --
        # those OOM'd MPS on this Mac's 16GB unified memory. Still clearly
        # deeper/wider than "dry" (enough for the depth-dependent ablations
        # to show real instability), just sized to actually fit.
        "ctx_len": 512,
        "emb_dim": 384,
        "n_heads": 6,
        "n_layers": 8,
        # Measured ~18.6s/step on this Mac at batch_size=8/grad_accum=15.
        # 300 steps (~1.5hr/run) was too short for cosine decay to show its
        # benefit or for fragile ablations (no-attn-scale, no-residual) to
        # diverge -- bumped to 1000 steps (~5.2hr/run, ~31hr for the 6
        # full-profile sweep entries) for real signal without a multi-day
        # commitment.
        "warmup_steps": 100,
        "max_steps": 1000,
        "eval_freq": 250,
        "eval_iters": 20,
        "batch_size": 8,
        "grad_accum_steps": 15,
    },
    "dry": {
        "ctx_len": 256,
        "emb_dim": 128,
        "n_heads": 4,
        "n_layers": 4,
        "warmup_steps": 20,
        "max_steps": 200,
        "eval_freq": 50,
        "eval_iters": 10,
        "batch_size": 8,
        "grad_accum_steps": 4,
    },
}
DROP_RATE: float = 0.0
BIAS: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small GPT on FineWeb-edu.")

    parser.add_argument("--profile", choices=["dry", "full"], default="dry")

    # Naming/orchestration -- used by sweep.py, harmless for a manual run.
    parser.add_argument("--run-label", default="manual")
    parser.add_argument("--wandb-group", default=None)

    # Optimization pitfalls.
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override the profile's step count, e.g. for a short diagnostic "
        "run before committing to a full-length comparison",
    )

    # Ablation switches (see docstrings on the flags below for what each
    # tests).
    parser.add_argument(
        "--attention-impl",
        choices=["sdpa", "manual", "manual_no_scale"],
        default="sdpa",
        help="manual_no_scale drops the 1/sqrt(d_k) scaling to show why it exists",
    )
    parser.add_argument(
        "--position-encoding",
        choices=["learned", "rope"],
        default="learned",
    )
    parser.add_argument(
        "--norm-placement",
        choices=["pre", "post"],
        default="pre",
        help="pre (GPT-2) vs post (original Transformer) LN ordering",
    )
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument(
        "--weight-decay-mode",
        choices=["decoupled", "l2"],
        default="decoupled",
    )
    parser.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument(
        "--n-heads",
        type=int,
        default=None,
        help="override the profile's head count -- shrinking it (same emb_dim) "
        "raises head_dim, for stress-testing attention scaling",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="override the profile's layer count -- raising it stress-tests "
        "depth-dependent effects like post-norm instability",
    )
    parser.add_argument(
        "--n-kv-heads",
        type=int,
        default=None,
        help="GQA group count (default = no-op MHA, one KV head per Q head)",
    )
    parser.add_argument(
        "--ablate-no-residual",
        action="store_true",
        help="show the residual stream is load-bearing",
    )
    parser.add_argument(
        "--ablate-no-norm",
        action="store_true",
        help="show norm's stabilizing effect at depth",
    )
    parser.add_argument(
        "--ablate-no-weight-tying",
        action="store_true",
        help="show the effect of tying tok_emb/out_head",
    )
    parser.add_argument(
        "--residual-scale-init",
        action="store_true",
        help="GPT-2/nanoGPT 1/sqrt(2*n_layers) residual-branch init",
    )

    # Checkpointing -- off by default; sweep runs are observational only.
    parser.add_argument("--checkpoint-path", default=None)

    args = parser.parse_args()

    if args.warmup_steps is None:
        args.warmup_steps = PROFILE_DEFAULTS[args.profile]["warmup_steps"]
    if args.max_steps is None:
        args.max_steps = PROFILE_DEFAULTS[args.profile]["max_steps"]
    if args.n_heads is None:
        args.n_heads = PROFILE_DEFAULTS[args.profile]["n_heads"]
    if args.n_layers is None:
        args.n_layers = PROFILE_DEFAULTS[args.profile]["n_layers"]
    if args.n_kv_heads is None:
        args.n_kv_heads = args.n_heads

    return args


@dataclass
class GPTConfig:
    vocab_size: int
    ctx_len: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float = DROP_RATE
    bias: bool = BIAS


@dataclass
class TokenizerState:
    pat_str: str
    mergeable_ranks: dict[bytes, int]
    special_tokens: dict[str, int]
    eot_token_id: int


class GzipTokens:
    # Array-like wrapper around a rapidgzip random-access handle.
    def __init__(
        self, gz_path: str, index_path: str, parallelization: int, dtype=TOKENS_DTYPE
    ):
        self.gz_path = gz_path
        self.index_path = index_path
        self.parallelization = parallelization
        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize
        self._file = None
        self._len: int | None = None
        # Open eagerly in the parent so length is known before any worker fork.
        self._ensure_open()

    def _ensure_open(self) -> None:
        if self._file is not None:
            return
        self._file = rapidgzip.open(self.gz_path, parallelization=self.parallelization)
        self._file.import_index(self.index_path)
        self._file.seek(0, 2)
        self._len = self._file.tell() // self.itemsize

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __len__(self) -> int:
        if self._len is None:
            self._ensure_open()
        return self._len

    def __getitem__(self, key: slice) -> np.ndarray:
        self._ensure_open()
        start, stop, step = key.indices(self._len)
        assert step == 1, "GzipTokens only supports contiguous slices"
        self._file.seek(start * self.itemsize)
        raw = self._file.read((stop - start) * self.itemsize)
        return np.frombuffer(raw, dtype=self.dtype)

    def close(self) -> None:
        # rapidgzip warns (and can abort the process) if its background
        # threads are still alive at Python interpreter shutdown, so this
        # must be called explicitly rather than left to __del__/GC timing.
        if self._file is not None:
            self._file.close()
            self._file = None


class GPTDataset(Dataset):
    def __init__(self, ids, max_length, stride):
        self.ids = ids
        self.max_length = max_length
        # Lazily indexed: works for a Python list or an np.memmap without
        # materializing every window up front (needed for corpus-scale token files).
        self.starts = range(0, len(ids) - max_length, stride)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        input_chunk = self.ids[start : start + self.max_length]
        target_chunk = self.ids[start + 1 : start + self.max_length + 1]
        return (
            torch.tensor(input_chunk, dtype=torch.long),
            torch.tensor(target_chunk, dtype=torch.long),
        )


def make_dataloader(
    ids, batch_size, max_length, stride, shuffle, drop_last, num_workers=0
):
    dataset = GPTDataset(ids, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=False,  # only meaningful for CUDA, which this script never uses
    )

    return dataloader


def build_encoding(state: TokenizerState) -> tiktoken.Encoding:
    return tiktoken.Encoding(
        name="dataset_tokenizer",
        pat_str=state.pat_str,
        mergeable_ranks=state.mergeable_ranks,
        special_tokens=state.special_tokens,
    )


def load_tokenizer_state(path: str):
    with open(path, "rb") as f:
        tokenizer_state = pickle.load(f)
    encoding = build_encoding(tokenizer_state)
    return encoding, tokenizer_state.eot_token_id, tokenizer_state


def ensure_token_indices() -> None:
    parallelization = os.cpu_count()
    if not os.path.exists(TRAIN_TOKENS_INDEX_PATH):
        with rapidgzip.open(TRAIN_TOKENS_GZ_PATH, parallelization=parallelization) as f:
            f.export_index(TRAIN_TOKENS_INDEX_PATH)
    if not os.path.exists(VAL_TOKENS_INDEX_PATH):
        with rapidgzip.open(VAL_TOKENS_GZ_PATH, parallelization=parallelization) as f:
            f.export_index(VAL_TOKENS_INDEX_PATH)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ctx_len,
        dropout,
        num_heads,
        num_kv_heads,
        attention_impl,
        position_encoding,
        bias=False,
    ):
        super().__init__()
        assert dim_out % num_heads == 0, "dim_out must be divisible by n_heads"

        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.attention_impl = attention_impl

        # GQA: num_kv_heads < num_heads shares each K/V head across several
        # query heads, to see the quality/compute tradeoff vs. standard MHA.
        self.num_kv_heads = num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads"
        )
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        kv_dim = self.num_kv_heads * self.head_dim

        self.W_query = nn.Linear(dim_in, dim_out, bias=bias)
        self.W_key = nn.Linear(dim_in, kv_dim, bias=bias)
        self.W_value = nn.Linear(dim_in, kv_dim, bias=bias)
        self.out_proj = nn.Linear(dim_out, dim_out, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # Needed by both the manual attention path and get_attention_weights
        # (which forces the manual path even when attention_impl=sdpa).
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(ctx_len, ctx_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

        # RoPE: rotate Q/K by position instead of adding a learned pos_emb at
        # the input, to see how relative-position encoding changes attention.
        self.use_rope = position_encoding == "rope"
        if self.use_rope:
            assert self.head_dim % 2 == 0, "RoPE requires an even head_dim"
            half_dim = self.head_dim // 2
            inv_freq = 1.0 / (
                10000 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
            )
            freqs = torch.outer(torch.arange(ctx_len, dtype=torch.float32), inv_freq)
            self.register_buffer("rope_cos", freqs.cos(), persistent=False)
            self.register_buffer("rope_sin", freqs.sin(), persistent=False)

    def _apply_rope(self, q, k):
        seq_len = q.size(-2)
        cos = self.rope_cos[:seq_len].to(q.dtype)[None, None, :, :]
        sin = self.rope_sin[:seq_len].to(q.dtype)[None, None, :, :]

        def rotate(x):
            x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return rotate(q), rotate(k)

    def forward(self, x, return_weights=False):
        batch_size, num_tokens, dim_in = x.shape

        keys = self.W_key(x)  # Shape: (batch_size, num_tokens, num_kv_heads * head_dim)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.reshape(batch_size, num_tokens, self.num_kv_heads, self.head_dim)
        values = values.reshape(
            batch_size, num_tokens, self.num_kv_heads, self.head_dim
        )
        queries = queries.reshape(batch_size, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.num_kv_groups > 1:
            # Repeat each KV head contiguously so kv-head i maps to query
            # heads [i*groups, (i+1)*groups) -- the standard GQA convention.
            keys = keys.repeat_interleave(self.num_kv_groups, dim=1)
            values = values.repeat_interleave(self.num_kv_groups, dim=1)

        if self.use_rope:
            queries, keys = self._apply_rope(queries, keys)

        attn_weights = None
        # sdpa can't expose weights, so return_weights forces the manual path
        # regardless of the globally configured attention_impl.
        use_manual = self.attention_impl != "sdpa" or return_weights
        if use_manual:
            scale = (
                1.0
                if self.attention_impl == "manual_no_scale"
                else 1.0 / math.sqrt(self.head_dim)
            )
            attn_scores = (queries @ keys.transpose(-2, -1)) * scale
            mask = self.causal_mask[:num_tokens, :num_tokens]
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out_embeds = (attn_weights @ values).transpose(1, 2)
        else:
            dropout_p = self.dropout.p if self.training else 0.0
            out_embeds = F.scaled_dot_product_attention(
                queries, keys, values, is_causal=True, dropout_p=dropout_p
            ).transpose(1, 2)

        # Combine heads, where self.dim_out = self.num_heads * self.head_dim
        out_embeds = out_embeds.reshape(batch_size, num_tokens, self.dim_out)
        out_embeds = self.out_proj(out_embeds)  # optional projection

        if return_weights:
            return out_embeds, attn_weights
        return out_embeds


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Constant kept as a Python float so nothing is re-materialized on the
        # host each call (and no CPU/CUDA tensor mismatch under torch.compile).
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.emb_dim, 4 * config.emb_dim),
            GELU(),
            nn.Linear(4 * config.emb_dim, config.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.att = MultiHeadAttention(
            dim_in=config.emb_dim,
            dim_out=config.emb_dim,
            ctx_len=config.ctx_len,
            num_heads=config.n_heads,
            num_kv_heads=args.n_kv_heads,
            attention_impl=args.attention_impl,
            position_encoding=args.position_encoding,
            dropout=config.drop_rate,
            bias=config.bias,
        )
        self.ff = FeedForward(config)
        self.ablate_no_norm = args.ablate_no_norm
        self.norm_placement = args.norm_placement
        self.ablate_no_residual = args.ablate_no_residual
        self.norm1 = nn.Identity() if args.ablate_no_norm else LayerNorm(config.emb_dim)
        self.norm2 = nn.Identity() if args.ablate_no_norm else LayerNorm(config.emb_dim)
        self.drop_attn = nn.Dropout(config.drop_rate)
        self.drop_ff = nn.Dropout(config.drop_rate)

    def _residual_step(self, x, norm, sublayer, drop):
        # norm_placement / ablate_no_residual / ablate_no_norm: strip each
        # piece of the pre-norm residual block in turn to see what it buys.
        if self.norm_placement == "post":
            out = drop(sublayer(x))
            return norm(out) if self.ablate_no_residual else norm(x + out)
        out = drop(sublayer(norm(x)))  # pre-norm (default, GPT-2 style)
        return out if self.ablate_no_residual else x + out

    def forward(self, x):
        x = self._residual_step(x, self.norm1, self.att, self.drop_attn)
        x = self._residual_step(x, self.norm2, self.ff, self.drop_ff)
        return x


class GPT(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        if args.position_encoding != "rope":
            self.pos_emb = nn.Embedding(config.ctx_len, config.emb_dim)
        self.drop_emb = nn.Dropout(config.drop_rate)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config, args) for _ in range(config.n_layers)]
        )

        self.final_norm = LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=config.bias)
        # Weight tying (GPT-2 style): input and output embeddings share one matrix,
        # which is why nn.Linear's [vocab_size, emb_dim] weight lines up with
        # nn.Embedding's. Roughly halves the embedding parameter count.
        if not args.ablate_no_weight_tying:
            self.out_head.weight = self.tok_emb.weight

        # nn.Embedding's default init has std=1.0; left alone, the tied output
        # projection above would blow up logit magnitude (loss >> ln(vocab_size)
        # at init). GPT-2-style N(0, init_std) keeps init sane for both roles.
        self.apply(self._init_weights)
        if args.residual_scale_init:
            self._apply_residual_scale_init(config.n_layers)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.args.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _apply_residual_scale_init(self, n_layers):
        # nanoGPT/GPT-2 trick: shrink each residual branch's output projection
        # by 1/sqrt(2*n_layers) at init so variance doesn't compound with depth.
        scale = 1.0 / math.sqrt(2 * n_layers)
        for block in self.trf_blocks:
            for module in (block.att.out_proj, block.ff.layers[2]):
                nn.init.normal_(module.weight, mean=0.0, std=self.args.init_std * scale)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        if self.args.position_encoding == "rope":
            x = tok_embeds  # position info is injected inside attention instead
        else:
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


def unwrap_model(model):
    # Never wrapped in DDP/torch.compile on this script; kept as a
    # passthrough so this stays structurally close to the Kaggle version.
    return model


def get_attention_weights(model, input_ids):
    # Offline utility for plotting attention maps -- forces the manual
    # softmax path (sdpa can't expose weights) and is never called from the
    # training loop, so the hot path pays nothing for this existing.
    base = unwrap_model(model)
    base.eval()
    weights_per_layer = []
    patches = []
    for block in base.trf_blocks:
        att = block.att
        original_forward = att.forward

        def make_patched(orig_fwd):
            def patched(x, return_weights=True):
                out, w = orig_fwd(x, return_weights=True)
                weights_per_layer.append(w)
                return out

            return patched

        att.forward = make_patched(original_forward)
        patches.append((att, original_forward))

    try:
        with torch.no_grad():
            base(input_ids)
    finally:
        for att_module, original_forward in patches:
            att_module.forward = original_forward

    return weights_per_layer


def save_checkpoint(
    path,
    model,
    optimizer,
    step: int,
    gpt_config: GPTConfig,
    tokenizer_state: TokenizerState,
    train_losses: list[float],
    val_losses: list[float],
    tokens_seen: list[int],
) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        rng_state: dict[str, object] = {"cpu": torch.get_rng_state()}

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "rng_state": rng_state,
                "gpt_config": asdict(gpt_config),
                "tokenizer_state": asdict(tokenizer_state),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "tokens_seen": tokens_seen,
            },
            path,
        )

        if wandb.run is not None:
            # A single artifact name (rather than one per step) means "latest"
            # always resolves to the most recent checkpoint.
            artifact = wandb.Artifact("checkpoint", type="model")
            artifact.add_file(path)
            wandb.log_artifact(artifact, aliases=[f"step-{step}"])


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    torch.set_rng_state(checkpoint["rng_state"]["cpu"].cpu())

    return (
        checkpoint["step"],
        checkpoint["train_losses"],
        checkpoint["val_losses"],
        checkpoint["tokens_seen"],
    )


def download_latest_wandb_checkpoint(path: str) -> bool:
    # Best-effort fallback for resume_if_available: a fresh environment has
    # no local checkpoint, but wandb might have one from a prior run. Returns
    # False (rather than raising) on any failure so a missing/misconfigured
    # wandb project just falls through to a cold start.
    project = os.getenv("WANDB_PROJECT", "femtochat")
    try:
        api = wandb.Api()
        entity = os.getenv("WANDB_ENTITY") or api.default_entity
        artifact = api.artifact(f"{entity}/{project}/checkpoint:latest", type="model")
        artifact_dir = artifact.download()
    except Exception:
        return False

    downloaded_path = next(Path(artifact_dir).glob("*.pt"), None)
    if downloaded_path is None:
        return False

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(downloaded_path), path)
    return True


class NoOpScaler:
    # Drop-in stand-in for torch.amp.GradScaler; this script never has a
    # real fp16 GradScaler since it never runs on CUDA.
    def __init__(self, enabled: bool):
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer) -> None:
        pass

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        pass


def make_scaler(device: torch.device):
    return NoOpScaler(enabled=USE_AMP)


def repeat_loader(loader, skip_batches: int = 0):
    # skip_batches fast-forwards the stream on resume so we don't re-serve the
    # earliest windows. Complete epochs are skipped for free by advancing the
    # epoch counter.
    batches_per_epoch = len(loader)
    epoch = skip_batches // batches_per_epoch if skip_batches else 0
    remainder = skip_batches % batches_per_epoch if skip_batches else 0
    while True:
        for batch_index, batch in enumerate(loader):
            if batch_index < remainder:
                continue
            yield batch
        remainder = 0  # only the resumed epoch is partially skipped
        epoch += 1


def get_lr(step: int, args, max_steps: int) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    if args.lr_schedule == "constant":
        return args.lr
    if step > max_steps:
        return args.min_lr
    decay_ratio = (step - args.warmup_steps) / max(1, max_steps - args.warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


def train_micro_batch(
    model,
    scaler,
    input_batch,
    target_batch,
    device,
    grad_accum_steps,
    weight_decay_mode,
    decayable_params=None,
) -> float:
    model.train()
    # non_blocking=True is a safe no-op on mps/cpu (only overlaps with compute
    # when the source tensor is pinned, which it never is here).
    with torch.autocast(
        device_type=device.type, dtype=AMP_DTYPE, enabled=scaler.is_enabled()
    ):
        logits = model(input_batch.to(device, non_blocking=True))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_batch.to(device, non_blocking=True).reshape(-1),
        )
        if weight_decay_mode == "l2" and decayable_params is not None:
            # Classic L2-in-loss weight decay, to compare against AdamW's
            # decoupled decay.
            l2 = sum(p.pow(2).sum() for p in decayable_params)
            loss = loss + 0.5 * WEIGHT_DECAY * l2
    scaler.scale(loss / grad_accum_steps).backward()

    return loss.item()


def evaluate(model, scaler, val_loader, device, eval_iters: int) -> float:
    model.eval()
    val_loss = 0.0
    num_batches = min(eval_iters, len(val_loader))
    with torch.no_grad():
        for i, (val_input, val_target) in enumerate(val_loader):
            if i >= num_batches:
                break
            # Mirror training precision so eval loss is measured on the same
            # numerics the model is optimized under.
            with torch.autocast(
                device_type=device.type, dtype=AMP_DTYPE, enabled=scaler.is_enabled()
            ):
                val_logits = model(val_input.to(device, non_blocking=True))
                batch_val_loss = F.cross_entropy(
                    val_logits.reshape(-1, val_logits.size(-1)),
                    val_target.to(device, non_blocking=True).reshape(-1),
                ).item()
            val_loss += batch_val_loss
    return val_loss / num_batches


def compute_update_ratios(model, lr):
    # (lr * grad_norm) / param_norm per layer: the classic "how big is this
    # step relative to the weight itself" diagnostic (Karpathy's ~1e-3 rule of
    # thumb) for spotting dead/exploding layers early in training.
    base = unwrap_model(model)
    named = {
        "tok_emb": base.tok_emb.weight,
        "block_0": base.trf_blocks[0].ff.layers[0].weight,
        "block_last": base.trf_blocks[-1].ff.layers[0].weight,
    }
    if base.out_head.weight is not base.tok_emb.weight:
        named["out_head"] = base.out_head.weight

    ratios = {}
    for name, param in named.items():
        if param.grad is None:
            continue
        ratio = (lr * param.grad.norm()) / (param.data.norm() + 1e-12)
        ratios[f"update_ratio/{name}"] = ratio.item()
    return ratios


def run_optimizer_step(
    model,
    optimizer,
    scaler,
    train_iter,
    device,
    step,
    args,
    max_steps,
    grad_accum_steps,
    decayable_params=None,
):
    lr = get_lr(step, args, max_steps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.zero_grad()
    accumulated_loss = 0.0
    step_tokens = 0
    for micro_step in range(grad_accum_steps):
        input_batch, target_batch = next(train_iter)
        accumulated_loss += train_micro_batch(
            model,
            scaler,
            input_batch,
            target_batch,
            device,
            grad_accum_steps,
            args.weight_decay_mode,
            decayable_params,
        )
        step_tokens += input_batch.numel()
    accumulated_loss /= grad_accum_steps

    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    update_ratios = compute_update_ratios(model, lr)
    scaler.step(optimizer)
    scaler.update()

    return accumulated_loss, step_tokens, lr, grad_norm, update_ratios


def evaluate_and_checkpoint(
    model,
    optimizer,
    scaler,
    val_loader,
    encoding,
    eot_token_id,
    gpt_config: GPTConfig,
    device: torch.device,
    tokenizer_state: TokenizerState,
    checkpoint_path: str,
    step_num: int,
    train_loss: float,
    total_tokens: int,
    train_losses: list[float],
    val_losses: list[float],
    tokens_seen: list[int],
    use_wandb: bool,
    writer: SummaryWriter,
    eval_iters: int,
) -> tuple[float, str]:
    train_losses.append(train_loss)
    tokens_seen.append(total_tokens)

    val_loss = evaluate(model, scaler, val_loader, device, eval_iters)
    val_losses.append(val_loss)

    output_text = generate_text(
        model=model,
        encoding=encoding,
        eot_token_id=eot_token_id,
        prompt=VAL_PROMPT,
        max_tokens=MAX_SAMPLE_TOKENS,
        ctx_len=gpt_config.ctx_len,
        device=device,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )

    if use_wandb:
        wandb.log({"val/loss": val_loss, "val/sample": output_text}, step=step_num)
    writer.add_scalar("val/loss", val_loss, step_num)
    writer.add_text("val/sample", output_text, step_num)

    save_checkpoint(
        checkpoint_path,
        model=unwrap_model(model),
        optimizer=optimizer,
        step=step_num,
        gpt_config=gpt_config,
        tokenizer_state=tokenizer_state,
        train_losses=train_losses,
        val_losses=val_losses,
        tokens_seen=tokens_seen,
    )

    return val_loss, output_text


def run_training(
    model,
    optimizer,
    scaler,
    train_loader,
    val_loader,
    encoding,
    eot_token_id,
    gpt_config: GPTConfig,
    device: torch.device,
    tokenizer_state: TokenizerState,
    checkpoint_path: str,
    args,
    max_steps: int,
    eval_freq: int,
    eval_iters: int,
    grad_accum_steps: int,
    use_wandb: bool = False,
    start_step: int = 0,
    train_losses: list[float] | None = None,
    val_losses: list[float] | None = None,
    tokens_seen: list[int] | None = None,
    decayable_params=None,
    writer: SummaryWriter | None = None,
):
    train_losses = train_losses if train_losses is not None else []
    val_losses = val_losses if val_losses is not None else []
    tokens_seen = tokens_seen if tokens_seen is not None else []
    total_tokens = tokens_seen[-1] if tokens_seen else 0
    train_iter = repeat_loader(train_loader, skip_batches=start_step * grad_accum_steps)

    for step in range(start_step, max_steps):
        accumulated_loss, step_tokens, lr, grad_norm, update_ratios = (
            run_optimizer_step(
                model,
                optimizer,
                scaler,
                train_iter,
                device,
                step,
                args,
                max_steps,
                grad_accum_steps,
                decayable_params,
            )
        )
        total_tokens += step_tokens
        step_num = step + 1

        if use_wandb:
            wandb.log(
                {
                    "train/loss": accumulated_loss,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "train/tokens_seen": total_tokens,
                    "train/progress": step_num / max_steps * 100,
                    **update_ratios,
                },
                step=step_num,
            )

        writer.add_scalar("train/loss", accumulated_loss, step_num)
        writer.add_scalar("train/lr", lr, step_num)
        writer.add_scalar("train/grad_norm", grad_norm, step_num)
        writer.add_scalar("train/tokens_seen", total_tokens, step_num)
        for name, value in update_ratios.items():
            writer.add_scalar(name, value, step_num)

        if step_num % eval_freq == 0 or step_num == max_steps:
            evaluate_and_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                val_loader=val_loader,
                encoding=encoding,
                eot_token_id=eot_token_id,
                gpt_config=gpt_config,
                device=device,
                tokenizer_state=tokenizer_state,
                checkpoint_path=checkpoint_path,
                step_num=step_num,
                train_loss=accumulated_loss,
                total_tokens=total_tokens,
                train_losses=train_losses,
                val_losses=val_losses,
                tokens_seen=tokens_seen,
                use_wandb=use_wandb,
                writer=writer,
                eval_iters=eval_iters,
            )

    return train_losses, val_losses, tokens_seen


def sample_ids(
    model, ids, eot_token_id, max_tokens, ctx_len, temperature=0.0, top_k=None
):
    for _ in range(max_tokens):
        # Crop current context if it exceeds the supported context size
        ids_cond = ids[:, -ctx_len:]
        with torch.no_grad():
            logits = model(ids_cond)

        logits = logits[:, -1, :]

        if temperature > 0:
            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits / temperature, dim=-1)
            id_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)
        else:
            id_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        ids = torch.cat((ids, id_next), dim=1)  # (batch, n_tokens+1)

        # Early stop only makes sense for a single sequence; for a batch,
        # different rows hit EOT at different steps, so keep generating.
        if ids.size(0) == 1 and id_next.item() == eot_token_id:
            break

    return ids


def generate_text(
    model,
    encoding,
    eot_token_id,
    prompt,
    max_tokens,
    ctx_len,
    device,
    temperature=0.0,
    top_k=None,
):
    model.eval()
    input_ids = torch.tensor(
        [eot_token_id] + encoding.encode_ordinary(prompt)
    ).unsqueeze(0)

    output_ids = sample_ids(
        model=model,
        ids=input_ids.to(device),
        eot_token_id=eot_token_id,
        max_tokens=max_tokens,
        ctx_len=ctx_len,
        temperature=temperature,
        top_k=top_k,
    )

    return encoding.decode(output_ids.squeeze(0).tolist())


def init_wandb(model, gpt_config: GPTConfig, args, run_name: str, eval_freq: int) -> None:
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "femtochat"),
        entity=os.getenv("WANDB_ENTITY"),
        name=run_name,
        group=args.wandb_group,
        mode="offline",
        config={
            **asdict(gpt_config),
            **vars(args),
            "device_type": DEVICE_TYPE,
        },
    )
    wandb.watch(model, log="gradients", log_freq=eval_freq)


def build_dataloaders(train_ids, val_ids, gpt_config: GPTConfig, batch_size: int):
    train_loader = make_dataloader(
        train_ids,
        batch_size=batch_size,
        max_length=gpt_config.ctx_len,
        stride=gpt_config.ctx_len,
        shuffle=True,
        drop_last=True,
    )
    val_loader = make_dataloader(
        val_ids,
        batch_size=batch_size,
        max_length=gpt_config.ctx_len,
        stride=gpt_config.ctx_len,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


def resume_if_available(checkpoint_path, profile, model, optimizer, device):
    if checkpoint_path is None:
        return 0, [], [], []

    if not os.path.exists(checkpoint_path):
        # Only the "full" profile's checkpoints are meant to round-trip through
        # wandb -- pulling one into a "dry" run would both hang on a slow/offline
        # connection and crash on load_state_dict (different model shapes).
        if profile == "full" and download_latest_wandb_checkpoint(checkpoint_path):
            print(f"No local checkpoint; downloaded latest from wandb to {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            return 0, [], [], []

    start_step, train_losses, val_losses, tokens_seen = load_checkpoint(
        checkpoint_path, model, optimizer, device
    )
    print(f"Resumed from {checkpoint_path} at step {start_step}")
    return start_step, train_losses, val_losses, tokens_seen


def main() -> None:
    args = parse_args()
    profile_defaults = PROFILE_DEFAULTS[args.profile]

    device = torch.device(DEVICE_TYPE)
    use_wandb = True
    torch.manual_seed(SEED)

    ensure_token_indices()

    gzip_parallelization = os.cpu_count()
    print(f"Using {device} device (profile={args.profile}).")

    run_name = f"femtochat-{args.profile}-{args.run_label}-{time.strftime('%Y%m%d-%H%M%S')}"

    with contextlib.ExitStack() as stack:
        if use_wandb:
            stack.callback(wandb.finish)

        writer = SummaryWriter(log_dir=f"{OUTPUT_DIR}/tensorboard/{run_name}")
        stack.callback(writer.close)

        encoding, eot_token_id, tokenizer_state = load_tokenizer_state(
            TOKENIZER_STATE_PATH
        )

        train_ids = GzipTokens(
            TRAIN_TOKENS_GZ_PATH,
            TRAIN_TOKENS_INDEX_PATH,
            parallelization=gzip_parallelization,
        )
        stack.callback(train_ids.close)
        val_ids = GzipTokens(
            VAL_TOKENS_GZ_PATH,
            VAL_TOKENS_INDEX_PATH,
            parallelization=gzip_parallelization,
        )
        stack.callback(val_ids.close)

        gpt_config = GPTConfig(
            vocab_size=encoding.n_vocab,
            ctx_len=profile_defaults["ctx_len"],
            emb_dim=profile_defaults["emb_dim"],
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        )
        model = GPT(gpt_config, args).to(device)

        if use_wandb:
            init_wandb(model, gpt_config, args, run_name, profile_defaults["eval_freq"])

        optimizer_kwargs = (
            {"momentum": 0.9} if args.optimizer == "sgd" else {"betas": BETAS}
        )
        if args.weight_decay_mode == "decoupled":
            optimizer_kwargs["weight_decay"] = WEIGHT_DECAY
        optimizer_cls = torch.optim.SGD if args.optimizer == "sgd" else torch.optim.AdamW
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, **optimizer_kwargs)

        decayable_params = (
            [p for p in model.parameters() if p.dim() >= 2]
            if args.weight_decay_mode == "l2"
            else None
        )

        start_step, train_losses, val_losses, tokens_seen = resume_if_available(
            args.checkpoint_path, args.profile, model, optimizer, device
        )

        scaler = make_scaler(device)
        train_loader, val_loader = build_dataloaders(
            train_ids, val_ids, gpt_config, profile_defaults["batch_size"]
        )

        run_training(
            model,
            optimizer,
            scaler,
            train_loader,
            val_loader,
            encoding,
            eot_token_id,
            gpt_config,
            device,
            tokenizer_state,
            args.checkpoint_path,
            args,
            max_steps=args.max_steps,
            eval_freq=profile_defaults["eval_freq"],
            eval_iters=profile_defaults["eval_iters"],
            grad_accum_steps=profile_defaults["grad_accum_steps"],
            use_wandb=use_wandb,
            start_step=start_step,
            train_losses=train_losses,
            val_losses=val_losses,
            tokens_seen=tokens_seen,
            decayable_params=decayable_params,
            writer=writer,
        )

        print(f"Finished run {run_name!r}, checkpoint={args.checkpoint_path}")


if __name__ == "__main__":
    main()
