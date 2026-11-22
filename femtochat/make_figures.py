"""Render every metric plot used in REPORT.md from the TensorBoard event files."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- design tokens (dataviz reference palette, light mode) --------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
ORANGE = "#eb6834"

TB = "output/tensorboard"

# label -> tensorboard event dir (relative to TB)
RUNS = {
    "baseline": "sweep/baseline",
    "sgd-vs-adamw": "sweep/sgd-vs-adamw",
    "init-std-too-high": "sweep/init-std-too-high",
    "init-std-too-low": "sweep/init-std-too-low",
    "init-std-formula": "init_std_formula/formula_1-over-sqrt-embdim",
    "no-weight-tying": "sweep/no-weight-tying",
    "no-attn-scale": "sweep/no-attn-scale",
    "no-grad-clip": "sweep/no-grad-clip",
    "high-lr-no-warmup": "sweep/high-lr-no-warmup",
    "no-residual": "sweep/no-residual",
    "no-norm": "sweep/no-norm",
    "post-norm": "sweep/post-norm",
    "no-residual-scale-init": "sweep/no-residual-scale-init",
    "l2-weight-decay": "sweep/l2-weight-decay",
    "constant-lr": "sweep/constant-lr",
    "attn-scaled": "attention_scaling/scaled_headdim384_initstd05",
    "attn-unscaled": "attention_scaling/unscaled_headdim384_initstd05",
    "wvclip-neither": "warmup_vs_clipping/neither",
    "wvclip-clip-only": "warmup_vs_clipping/clip-only",
    "wvclip-warmup-only": "warmup_vs_clipping/warmup-only",
    "wvclip-both": "warmup_vs_clipping/both",
}

_cache: dict[str, dict] = {}


def load(label: str) -> dict:
    if label not in _cache:
        ea = EventAccumulator(os.path.join(TB, RUNS[label]), size_guidance={"scalars": 100000})
        ea.Reload()
        _cache[label] = {
            tag: ([e.step for e in ea.Scalars(tag)], [e.value for e in ea.Scalars(tag)])
            for tag in ea.Tags()["scalars"]
        }
    return _cache[label]


def style(ax, title):
    ax.set_title(title, color=INK, fontsize=11, pad=9)
    ax.set_xlabel("step", color=INK_2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)


def finish(fig, series_count):
    """Title row + one legend row under it, then trim the layout."""
    fig.legend(
        *fig.axes[0].get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.02, 0.925),
        frameon=False,
        fontsize=8.5,
        ncol=min(series_count, 6),
        labelcolor=INK_2,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])


def panel(ax, title, log=False, val=False):
    style(ax, title)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(("validation loss" if val else title.split(" (")[0].lower()), color=INK_2, fontsize=9)


DISPLAY = {"attn-scaled": "scaled", "attn-unscaled": "unscaled"}


def draw(ax, series, tag, markers=False, val=False, nudges=None, labels=False):
    nudges = nudges or {}
    entries = []
    for key, color in series:
        label = DISPLAY.get(key, key)
        steps, vals = load(key)[tag]
        ax.plot(
            steps,
            vals,
            color=color,
            linewidth=1.7,
            label=label,
            marker="o" if markers else None,
            markersize=3.5,
            solid_capstyle="round",
        )
        entries.append((label, steps, vals, color))
    if labels:
        for label, steps, vals, color in entries:
            ax.annotate(
                label,
                xy=(steps[-1], vals[-1]),
                xytext=(5, nudges.get(label, 0)),
                textcoords="offset points",
                color=color,
                fontsize=8,
                va="center",
            )
    return entries


def sweep_fig(fname, title, series, loss_log=False, labels=False, nudges=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    fig.suptitle(title, color=INK, fontsize=12, x=0.02, ha="left")
    panel(axes[0], "Train loss" + (" (log scale)" if loss_log else ""), log=loss_log)
    draw(axes[0], series, "train/loss", nudges=nudges, labels=labels)
    panel(axes[1], "Gradient norm (log scale)", log=True)
    draw(axes[1], series, "train/grad_norm", nudges=nudges, labels=labels)
    finish(fig, len(series))
    fig.savefig(f"figures/{fname}.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    os.makedirs("figures", exist_ok=True)

    # --- first sweep, dry profile: optimizer / init / tying ------------------
    sweep_fig(
        "sweep_optimizer_init",
        "Optimizer, initialization & weight tying (dry profile, 200 steps)",
        [
            ("baseline", BLUE),
            ("sgd-vs-adamw", RED),
            ("init-std-too-high", ORANGE),
            ("init-std-too-low", AQUA),
            ("no-weight-tying", GREEN),
            ("init-std-formula", VIOLET),
        ],
    )

    # --- first sweep, full profile: architecture -----------------------------
    sweep_fig(
        "sweep_architecture_full",
        "Architecture (full profile, 1000 steps)",
        [
            ("post-norm", BLUE),
            ("no-norm", RED),
            ("no-residual", ORANGE),
            ("no-residual-scale-init", VIOLET),
        ],
        labels=True,
        nudges={"no-norm": 10, "post-norm": -10, "no-residual-scale-init": -10},
    )

    # --- first sweep, full profile: optimization dynamics --------------------
    # l2's reported *train* loss includes the penalty term (starts ~420), so the
    # fair loss comparison here is validation loss.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    fig.suptitle(
        "Weight decay & LR schedule (full profile, 1000 steps)",
        color=INK,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    series = [("post-norm", BLUE), ("l2-weight-decay", RED), ("constant-lr", AQUA)]
    panel(axes[0], "Validation loss", val=True)
    draw(axes[0], series, "val/loss", markers=True, labels=True)
    panel(axes[1], "Gradient norm (log scale)", log=True)
    draw(axes[1], series, "train/grad_norm", labels=True, nudges={"constant-lr": -10})
    finish(fig, len(series))
    fig.savefig("figures/sweep_dynamics_full.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # --- first sweep, dry profile: optimization dynamics ---------------------
    sweep_fig(
        "sweep_dynamics_dry",
        "Small-scale checks: attention scale, clipping & warmup (dry profile, 200 steps)",
        [
            ("baseline", BLUE),
            ("no-attn-scale", ORANGE),
            ("no-grad-clip", VIOLET),
            ("high-lr-no-warmup", RED),
        ],
    )

    # --- follow-up: attention scaling ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    fig.suptitle(
        "Attention scaling (full, head_dim=384, init_std=0.05)",
        color=INK,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    series = [("attn-scaled", BLUE), ("attn-unscaled", RED)]
    panel(axes[0], "Train loss")
    draw(axes[0], series, "train/loss", labels=True)
    panel(axes[1], "Gradient norm (log scale)", log=True)
    draw(axes[1], series, "train/grad_norm", labels=True)
    finish(fig, len(series))
    fig.savefig("figures/exp_attention_scaling.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # --- follow-up: warmup vs clipping --------------------------------------
    runs = [
        ("neither", load("wvclip-neither"), RED),
        ("clip only", load("wvclip-clip-only"), ORANGE),
        ("warmup only", load("wvclip-warmup-only"), BLUE),
        ("both", load("wvclip-both"), AQUA),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)
    fig.suptitle(
        "Gradient clipping vs warmup at LR=0.3 (48 layers, post-norm, init_std=0.1)",
        color=INK,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    style(axes[0], "Train loss (full run)")
    axes[0].set_ylabel("loss", color=INK_2, fontsize=9)
    for label, data, color in runs:
        axes[0].plot(*data["train/loss"], color=color, linewidth=1.8, label=label)

    style(axes[1], "Train loss (first 30 steps)")
    axes[1].set_ylabel("loss", color=INK_2, fontsize=9)
    for label, data, color in runs:
        steps, vals = data["train/loss"]
        axes[1].plot(steps[:30], vals[:30], color=color, linewidth=1.8, marker="o", markersize=3.5)
    axes[1].annotate(
        "no warmup: spike to 53",
        xy=(3, 53.4),
        xytext=(10, -4),
        textcoords="offset points",
        color=RED,
        fontsize=8,
        va="top",
    )
    axes[1].annotate(
        "warmup: no spike",
        xy=(9, 9.0),
        xytext=(12, -4),
        textcoords="offset points",
        color=BLUE,
        fontsize=8,
        va="top",
    )

    style(axes[2], "Gradient norm (first 30 steps, log scale)")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("grad norm", color=INK_2, fontsize=9)
    for label, data, color in runs:
        steps, vals = data["train/grad_norm"]
        axes[2].plot(steps[:30], vals[:30], color=color, linewidth=1.8, marker="o", markersize=3.5)
        axes[2].annotate(
            label,
            xy=(steps[:30][-1], vals[:30][-1]),
            xytext=(5, {"clip only": -9, "both": 9}.get(label, 0)),
            textcoords="offset points",
            color=color,
            fontsize=8,
            va="center",
        )

    finish(fig, 4)
    fig.savefig("figures/exp_warmup_vs_clip.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
    print("figures written to figures/")
