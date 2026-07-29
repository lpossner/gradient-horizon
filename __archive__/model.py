"""
Transformer model
"""

from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    sequence_len: int = 512
    vocab_size: int = 32768
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Combined QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.sequence_len, config.sequence_len)).view(
                1, 1, config.sequence_len, config.sequence_len
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Reshape: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # att = (Q @ K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply causal mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values
        # Shape:(B, n_head, T, head_dim)
        y = att @ v

        # Reshape: (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    """
    Feed-forward network.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    Transformer decoder.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                # Token embeddings
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                # Learned positional embeddings
                "wpe": nn.Embedding(config.sequence_len, config.n_embd),
                # Dropout after embeddings
                "drop": nn.Dropout(config.dropout),
                # Transformer blocks
                "h": nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layer)]
                ),
                # Final layer norm
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )

        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embeddings and output projection
        self.lm_head.weight = self.transformer.wte.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Standard initialization following GPT-2.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            idx: Input token indices, shape (B, T)
            targets: Target token indices for loss computation, shape (B, T)

        Returns:
            If targets is None: logits of shape (B, T, vocab_size)
            If targets is provided: cross-entropy loss
        """
        B, T = idx.size()
        assert T <= self.config.sequence_len, (
            f"Sequence length {T} exceeds maximum {self.config.sequence_len}"
        )

        # Create position indices
        # (T,)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Token + positional embeddings
        # (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)
        # (T, n_embd)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Project to vocabulary
        # (B, T, vocab_size)
        logits = self.lm_head(x)

        # Compute loss if targets provided
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return loss

        return logits

    def get_optimizer(
        self,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.AdamW:
        """
        Get AdamW optimizer with weight decay only on 2D parameters.

        Args:
            learning_rate: Learning rate for the optimizer.
            weight_decay: Weight decay coefficient.
            betas: Adam betas (beta1, beta2).

        Returns:
            Configured AdamW optimizer.
        """
        # Separate parameters into those with and without weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to biases, LayerNorm, or embeddings
            if param.ndim < 2 or "ln" in name or "wpe" in name or "wte" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            idx: Conditioning sequence, shape (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = neutral, <1 = sharper, >1 = softer)
            top_k: If set, only sample from top k most likely tokens

        Returns:
            Generated sequence including input, shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to sequence length if needed
            idx_cond = (
                idx
                if idx.size(1) <= self.config.sequence_len
                else idx[:, -self.config.sequence_len :]
            )

            # Forward pass
            logits = self(idx_cond)

            # Get logits for last position and apply temperature
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
