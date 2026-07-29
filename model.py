import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, ctx_len, dropout, num_heads, bias=False):
        super().__init__()
        assert dim_out % num_heads == 0, "dim_out must be divisible by n_heads"

        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads

        self.W_query = nn.Linear(dim_in, dim_out, bias=bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=bias)
        self.out_proj = nn.Linear(dim_out, dim_out, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, dim_in = x.shape

        keys = self.W_key(x)  # Shape: (batch_size, num_tokens, dim_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (batch_size, num_tokens, dim_out) -> (batch_size, num_tokens, num_heads, head_dim)
        keys = keys.reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.reshape(batch_size, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (batch_size, num_tokens, num_heads, head_dim) -> (batch_size, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (batch_size, num_tokens, num_heads, head_dim)
        out_embeds = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.dim_out = self.num_heads * self.head_dim
        out_embeds = out_embeds.reshape(batch_size, num_tokens, self.dim_out)
        out_embeds = self.out_proj(out_embeds)  # optional projection

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
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
            dim_in=config["emb_dim"],
            dim_out=config["emb_dim"],
            ctx_len=config["ctx_len"],
            num_heads=config["n_heads"],
            dropout=config["drop_rate"],
            bias=config["bias"],
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.drop_attn = nn.Dropout(config["drop_rate"])
        self.drop_ff = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # Attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_attn(x)
        x = x + shortcut  # Add the original input back

        # Feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_ff(x)
        x = x + shortcut  # Add the original input back

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["ctx_len"], config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )

        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(
            config["emb_dim"], config["vocab_size"], bias=config["bias"]
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
