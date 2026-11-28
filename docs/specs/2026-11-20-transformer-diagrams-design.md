# Transformer diagrams: explicit residual paths and positional embeddings

Date: 2026-11-20
Post: `_posts/2026-11-15-architechture-of-femtochat.md`
Target: the two ` ```mermaid ` blocks in that post

## Intent

The post's prose explains residuals and positional embeddings, but neither
diagram shows them. Both blocks draw the Transformer as a single chain, so:

- **Residual connections are invisible.** Each block has an "Add" box with
  one visible input. Nothing shows where the second addend comes from, which
  is the whole point of a skip connection.
- **Positional information is hidden.** It sits inside a combined
  `Token embedding + positional encoding` node, so the reader cannot see that
  positional information is a *separate* signal that gets added in.

Goal: make both visible, in both figures, so the diagrams carry the same
information as the surrounding text. This is a teaching post building toward
an implementation, so explicitness beats compactness.

## Findings from rendering the current diagrams

Rendered against the pinned `mermaid@11.17.2`, through the same
`mermaid.init()` path `post.min.js` uses, in both themes.

**Two real defects in figure 2**, both worth clearing while the block is open:

1. **The cluster title collides with the first node.** `Decoder block ·
   repeated N times` is drawn inside the block's top edge and the
   embedding→block arrow descends through the text onto it. Measured cause:
   mermaid left-aligns the cluster label, and a title this long is nearly as
   wide as the cluster box, so a centre-entering edge always crosses it. No
   margin value fixes it — the fix has to be a shorter title, and/or routing
   the entry edge to the cluster instead of the node.
2. **The cluster title is illegible in dark mode.** The post's `%%{init}%%`
   sets `theme:'base'`, which overrides Chirpy's light/dark toggle entirely
   (verified: the two themes render byte-identical CSS). The `base` theme
   draws cluster labels in dark navy, which is invisible on the dark
   background. This is the failure mode `mermaid-must-be-theme-aware` warns
   about, and the current `'background':'#ffffff'` is the same bug.

## Approaches considered (residual explicitness)

- **A — dashed skip edges only.** Dotted bypass edges, `Add (residual)`
  labels unchanged. Minimal, but the reader still infers the operation.
- **B — skip edges + formula.** Skip edges, and the Add boxes name the
  operation: `x + attn(ln x)`.
- **C — explicit residual stream.** A small `x` node inside the block that
  both bypasses originate from, plus skip edges, so the stream reads as a
  rail. *Chosen.*

C was chosen over A and B because it makes the stream itself the visible
subject rather than an annotation on a chain — which is what the surrounding
prose is actually about.

## Design

**Figure 1 (encoder–decoder).** Keep the existing two-column `TB` layout and
the cluster-level `ENC ==>|keys, values| DEC` edge. In each of the Encoder
and Decoder columns:

- Split `… embedding + positional encoding` into two nodes feeding one `Add`.
- Add an `x` node after that `Add`; the sublayer bypasses start from it.
- Draw each bypass as a dotted edge into its `Add & Norm`.
- Labels stay `Add & Norm` (this figure is a reference reproduction of the
  paper's diagram, and `attn`/`ffn` notation would not match it).

**Figure 2 (autoregressive causal decoder).** Same treatment, with the
formula notation, since this is the figure the implementation tracks:

- Split the embedding node; `Token embedding` and `Positional embedding`
  both feed one `Add`.
- `x` node inside the block; bypasses `x -.-> a1` and `a1 -.-> a2`.
- Add boxes read `x + attn(ln x)` and `x + ffn(ln x)`.
- Shorten the cluster title to `Decoder block × N` and route the entry edge
  to the cluster (`ea --> BLK`) rather than the node, which clears the
  title collision.

**Shared styling, both figures.** The `%%{init}%%` directive becomes:

- `theme:'base'` with explicit theme variables — keep, so `classDef` fills
  (self-contained light chips) read on either background.
- Drop `'background':'#ffffff'`; use `'background':'transparent'`.
- `'clusterBkg':'transparent'` — **required**. Without it the `base` theme
  paints a solid pale-yellow slab that glares in dark mode.
- `'clusterLabelColor':'#6b7280'` — **required** for the cluster title to be
  legible on dark. `titleColor` does not do this; verified separately.
- Neutral `lineColor` / `clusterBorder` / `edgeLabelBackground` mid-greys.

Verified end state: both figures render correctly in both themes, with no
title collision and no white slab. `&times;` and `&middot;` in node text
render as `×` and `·`.

## Non-goals

- No change to the post's prose, headings, or the surrounding text.
- No change to the `mermaid` version pin or the theme.
- Figure 1 stays a faithful reproduction of the paper's diagram; it does
  not adopt the `attn`/`ffn` notation.

## Notes for review

- Verified with a local harness that mimics `post.min.js` exactly, so these
  renders match what readers get rather than approximating it.
