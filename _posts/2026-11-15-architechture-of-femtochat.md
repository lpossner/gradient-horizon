---
title: "Architecture of Femtochat"
date: 2026-11-15
categories: []
tags: []
description: "The architecture of Femtochat: an autoregressive Decoder in the style of GPT-2, NanoGPT, and Nanochat."
mermaid: true
math: true
---

This post is about the architecture of **Femtochat**. I aim to build a model that can follow simple instructions and generate coherent answers in English. The architecture of **Femtochat** is based on the Transformer architecture introduced by [Vaswani et al.][attention-is-all-you-need-arxiv]. The original Transformer architecture is a sequence-to-sequence model and consists of an Encoder and a Decoder that are connected by cross-attention layers in the Decoder. Both the Encoder and the Decoder are stacks of identical layers. A single Encoder layer and a single Decoder layer are shown below.

```mermaid
%%{init: {'theme':'base','themeVariables':{'lineColor':'#8b93a7','clusterBkg':'transparent','clusterBorder':'#8b93a7','edgeLabelBackground':'transparent','titleColor':'#6b7280','textColor':'#6b7280'},'flowchart':{'nodeSpacing':10,'rankSpacing':16,'padding':4,'subGraphTitleMargin':{'top':14,'bottom':6},'curve':'basis','useMaxWidth':true}}}%%
flowchart LR
    classDef compute fill:#e7ecfd,stroke:#4f5fd0,color:#111827
    classDef norm    fill:#e4f6ea,stroke:#3f9e5e,color:#111827

    subgraph EL["Encoder layer"]
        direction TB
        e1("Multi-head<br/>self-attention"):::compute
        e2("Add &amp; Norm"):::norm
        e3("Feed-forward<br/>network"):::compute
        e4("Add &amp; Norm"):::norm
        e1 --> e2 --> e3 --> e4
    end

    subgraph DL["Decoder layer"]
        direction TB
        d1("Masked multi-head<br/>self-attention"):::compute
        d2("Add &amp; Norm"):::norm
        d3("Multi-head<br/>cross-attention"):::compute
        d4("Add &amp; Norm"):::norm
        d5("Feed-forward<br/>network"):::compute
        d6("Add &amp; Norm"):::norm
        d1 --> d2 --> d3 --> d4 --> d5 --> d6
    end

    EL ~~~ DL
```

The full architecture stacks $N$ of these layers in the Encoder and $N$ of them in the Decoder.

```mermaid
%%{init: {'theme':'base','themeVariables':{'lineColor':'#8b93a7','clusterBkg':'transparent','clusterBorder':'#8b93a7','edgeLabelBackground':'transparent','titleColor':'#6b7280','textColor':'#6b7280'},'flowchart':{'nodeSpacing':10,'rankSpacing':20,'padding':4,'subGraphTitleMargin':{'top':14,'bottom':6},'curve':'basis','useMaxWidth':true}}}%%
flowchart LR
    classDef io      fill:#eef1f5,stroke:#9aa5b1,color:#111827
    classDef embed   fill:#fff3d6,stroke:#d9a13b,color:#111827
    classDef stack   fill:#e7ecfd,stroke:#4f5fd0,color:#111827
    classDef head    fill:#f4e8fb,stroke:#8b5cd6,color:#111827

    subgraph ENC["Encoder · N layers"]
        direction TB
        ei("Input tokens<br/>source sequence"):::io
        ee("Input embedding<br/>+ positional encoding"):::embed
        eb("⋮<br/>Transformer layer<br/>× N"):::stack
        eo("Encoder output"):::io
        ei --> ee --> eb --> eo
    end

    subgraph DEC["Decoder · N layers"]
        direction TB
        di("Output tokens<br/>shifted right"):::io
        de("Output embedding<br/>+ positional encoding"):::embed
        db("⋮<br/>Transformer layer<br/>+ cross-attention<br/>× N"):::stack
        dh("Linear + Softmax"):::head
        di --> de --> db --> dh
    end

    ENC ==>|keys, values| DEC
```

Most modern language models predict one token at a time and not a sequence. For this, they only use the Decoder part of the original Transformer architecture. The models predict the next token $t_{i+1}$ based on the sequence of preceding tokens $(t_1, \ldots, t_i)$. This is called autoregressive generation. Their self-attention mechanism lets the embedding $e_i$ of token $t_i$ attend only to the embeddings $e_1, \ldots, e_i$ of earlier tokens and itself. This is achieved by causal masking. Hence, most modern language models are autoregressive causal Decoder models. The architecture is shown below.

```mermaid
%%{init: {'theme':'base','themeVariables':{'lineColor':'#8b93a7','clusterBkg':'transparent','clusterBorder':'#8b93a7','edgeLabelBackground':'transparent','titleColor':'#6b7280','textColor':'#6b7280'},'flowchart':{'nodeSpacing':10,'rankSpacing':18,'padding':4,'subGraphTitleMargin':{'top':14,'bottom':6},'curve':'basis','useMaxWidth':true}}}%%
flowchart TB
    classDef io      fill:#eef1f5,stroke:#9aa5b1,color:#111827
    classDef embed   fill:#fff3d6,stroke:#d9a13b,color:#111827
    classDef stack   fill:#e7ecfd,stroke:#4f5fd0,color:#111827
    classDef norm    fill:#e4f6ea,stroke:#3f9e5e,color:#111827
    classDef head    fill:#f4e8fb,stroke:#8b5cd6,color:#111827

    ti("Input tokens<br/>prompt sequence"):::io
    te("Token embedding<br/>+ positional encoding"):::embed
    tb("⋮<br/>Transformer block<br/>LayerNorm → self-attention → Add<br/>LayerNorm → feed-forward → Add<br/>× N"):::stack
    ln("Final layer norm"):::norm
    lm("Linear · LM head"):::head
    out("Softmax<br/>next-token probabilities"):::head

    ti --> te --> tb --> ln --> lm --> out
```

Modern language models use many [variants][llm-architecture-gallery] of this architecture. In a first iteration, **Femtochat** uses the classical GPT-2-style architecture that was also used in [NanoGPT][nanogpt-github] and [Nanochat][nanochat-github].

[nanogpt-github]: https://github.com/karpathy/nanoGPT
[nanochat-github]: https://github.com/karpathy/nanochat/
[attention-is-all-you-need-arxiv]: https://arxiv.org/abs/1706.03762
[llm-architecture-gallery]: https://sebastianraschka.com/llm-architecture-gallery/
