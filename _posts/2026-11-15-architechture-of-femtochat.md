---
title: "Architecture of Femtochat"
date: 2026-11-15
categories: []
tags: []
description: "The architecture of Femtochat: an autoregressive Decoder in the style of GPT-2, NanoGPT, and Nanochat."
mermaid: true
math: true
---

This post is about the architecture of **Femtochat**. I aim to build a model that can follow simple instructions and generate coherent answers in English. The architecture of **Femtochat** is based on the Transformer architecture introduced by [Vaswani et al.][attention-is-all-you-need-arxiv]. The original Transformer architecture is a sequence-to-sequence model and consists of an Encoder and a Decoder that are connected by cross-attention layers in the Decoder. The architecture is shown below.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#ffffff','primaryColor':'#e7ecfd','primaryTextColor':'#111827','primaryBorderColor':'#4f5fd0','lineColor':'#4a5568','edgeLabelBackground':'#f4f5f7','clusterBkg':'#fffbf0','clusterBorder':'#d9a13b'}}}%%
flowchart TB
    classDef io      fill:#eef1f5,stroke:#9aa5b1,color:#111827
    classDef embed   fill:#fff3d6,stroke:#d9a13b,color:#111827
    classDef compute fill:#e7ecfd,stroke:#4f5fd0,stroke-width:1.5px,color:#111827
    classDef norm    fill:#e4f6ea,stroke:#3f9e5e,color:#111827
    classDef head    fill:#f4e8fb,stroke:#8b5cd6,stroke-width:1.5px,color:#111827

    subgraph ENC["Encoder · N layers"]
        direction TB
        ei["Input tokens<br/>source sequence"]:::io
        ee["Input embedding<br/>+ positional encoding"]:::embed
        eatt["Multi-head<br/>self-attention"]:::compute
        en1["Add &amp; Norm"]:::norm
        eff["Feed-forward<br/>network"]:::compute
        en2["Add &amp; Norm"]:::norm
        eo["Encoder output"]:::io
        ei --> ee --> eatt --> en1 --> eff --> en2 --> eo
    end

    subgraph DEC["Decoder · N layers"]
        direction TB
        di["Output tokens<br/>shifted right"]:::io
        de["Output embedding<br/>+ positional encoding"]:::embed
        datt["Masked multi-head<br/>self-attention"]:::compute
        dn1["Add &amp; Norm"]:::norm
        dca["Multi-head<br/>cross-attention"]:::compute
        dn2["Add &amp; Norm"]:::norm
        dff["Feed-forward<br/>network"]:::compute
        dn3["Add &amp; Norm"]:::norm
        dlin["Linear"]:::head
        dout["Softmax<br/>next-token probabilities"]:::head
        di --> de --> datt --> dn1 --> dca --> dn2 --> dff --> dn3 --> dlin --> dout
    end

    de ~~~ eo
    eo ==>|keys, values| dca
```

Most modern language models predict one token at a time and not a sequence. For this, they only use the Decoder part of the original Transformer architecture. The models predict the next token $t_{i+1}$ based on the sequence of preceding tokens $(t_1, \ldots, t_i)$. This is called autoregressive generation. Their self-attention mechanism lets the embedding $e_i$ of token $t_i$ attend only to the embeddings $e_1, \ldots, e_i$ of earlier tokens and itself. This is achieved by causal masking. Hence, most modern language models are autoregressive causal Decoder models. The architecture is shown below.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'#ffffff','primaryColor':'#e7ecfd','primaryTextColor':'#111827','primaryBorderColor':'#4f5fd0','lineColor':'#4a5568','edgeLabelBackground':'#f4f5f7','clusterBkg':'#fffbf0','clusterBorder':'#d9a13b'}}}%%
flowchart TB
    classDef io      fill:#eef1f5,stroke:#9aa5b1,color:#111827
    classDef embed   fill:#fff3d6,stroke:#d9a13b,color:#111827
    classDef compute fill:#e7ecfd,stroke:#4f5fd0,stroke-width:1.5px,color:#111827
    classDef norm    fill:#e4f6ea,stroke:#3f9e5e,color:#111827
    classDef head    fill:#f4e8fb,stroke:#8b5cd6,stroke-width:1.5px,color:#111827

    ti["Input tokens<br/>prompt sequence"]:::io
    te["Token embedding<br/>+ positional encoding"]:::embed

    subgraph BLK["Decoder block · repeated N times"]
        direction TB
        ln1["Layer norm"]:::norm
        att["Masked multi-head<br/>self-attention"]:::compute
        add1["Add (residual)"]:::norm
        ln2["Layer norm"]:::norm
        ff["Feed-forward<br/>network"]:::compute
        add2["Add (residual)"]:::norm
        ln1 --> att --> add1 --> ln2 --> ff --> add2
    end

    lnf["Final layer norm"]:::norm
    lin["Linear<br/>LM head"]:::head
    out["Softmax<br/>next-token probabilities"]:::head

    ti --> te --> ln1
    add2 --> lnf --> lin --> out
```

Modern language models use many [variants][llm-architecture-gallery] of this architecture. In a first iteration, **Femtochat** uses the classical GPT-2-style architecture that was also used in [NanoGPT][nanogpt-github] and [Nanochat][nanochat-github].

[nanogpt-github]: https://github.com/karpathy/nanoGPT
[nanochat-github]: https://github.com/karpathy/nanochat/
[attention-is-all-you-need-arxiv]: https://arxiv.org/abs/1706.03762
[llm-architecture-gallery]: https://sebastianraschka.com/llm-architecture-gallery/
