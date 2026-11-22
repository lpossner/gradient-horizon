import subprocess
import sys
import time

# Each entry names one lesson from a SOTA-training postmortem and the flag
# overrides that reproduce it against train.py's defaults (pre-norm AdamW,
# sdpa attention, learned position encoding, cosine LR, weight tying, no
# residual-scale-init). "profile" defaults to "dry" when omitted -- bump to
# "full" for anything whose effect only shows up with real depth/head_dim.
RUNS: list[dict] = [
    {
        "label": "baseline",
        "flags": {},
    },
    {
        "label": "no-attn-scale",
        "flags": {"attention_impl": "manual_no_scale"},
    },
    {
        "label": "sgd-vs-adamw",
        "flags": {"optimizer": "sgd"},
    },
    {
        "label": "no-residual",
        "profile": "full",
        "flags": {"ablate_no_residual": True},
    },
    {
        "label": "no-norm",
        "profile": "full",
        "flags": {"ablate_no_norm": True},
    },
    {
        "label": "post-norm",
        "profile": "full",
        "flags": {"norm_placement": "post"},
    },
    {
        "label": "no-residual-scale-init",
        "profile": "full",
        "flags": {"norm_placement": "post", "residual_scale_init": False},
    },
    {
        "label": "init-std-too-high",
        "flags": {"init_std": 0.5},
    },
    {
        "label": "init-std-too-low",
        "flags": {"init_std": 0.002},
    },
    {
        # 1/sqrt(emb_dim) for the dry profile's emb_dim=128 -- the fan-in
        # formula GPT-2's flat 0.02 only approximates for its own (much
        # wider) model sizes. Compare against "baseline" (std=0.02, same
        # profile) to see whether the formula converges faster.
        "label": "init-std-formula",
        "flags": {"init_std": 0.08838834764831845},
    },
    {
        "label": "l2-weight-decay",
        "profile": "full",
        "flags": {"weight_decay_mode": "l2"},
    },
    {
        "label": "constant-lr",
        "profile": "full",
        "flags": {"lr_schedule": "constant"},
    },
    {
        "label": "no-weight-tying",
        "flags": {"ablate_no_weight_tying": True},
    },
    {
        "label": "high-lr-no-warmup",
        "flags": {"lr": 5e-3, "warmup_steps": 0},
    },
    {
        "label": "no-grad-clip",
        "flags": {"grad_clip": 1e9},
    },
]


def build_argv(profile: str, run_label: str, wandb_group: str, flags: dict) -> list[str]:
    argv = [
        "train.py",
        "--profile",
        profile,
        "--run-label",
        run_label,
        "--wandb-group",
        wandb_group,
    ]
    for key, value in flags.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return argv


def main() -> None:
    sweep_id = time.strftime("sweep-%Y%m%d-%H%M%S")
    print(f"Starting sweep {sweep_id} ({len(RUNS)} runs)")

    results: list[tuple[str, bool]] = []
    for run in RUNS:
        label = run["label"]
        profile = run.get("profile", "dry")
        argv = build_argv(profile, label, sweep_id, run.get("flags", {}))

        print(f"\n=== [{sweep_id}] {label} (profile={profile}) ===")
        print(" ".join(argv))
        proc = subprocess.run([sys.executable, *argv])

        ok = proc.returncode == 0
        results.append((label, ok))
        if not ok:
            print(f"!!! run {label!r} failed with exit code {proc.returncode}, continuing")

    print(f"\n=== sweep {sweep_id} summary ===")
    for label, ok in results:
        print(f"{'OK  ' if ok else 'FAIL'}  {label}")

    num_failed = sum(1 for _, ok in results if not ok)
    if num_failed:
        print(f"\n{num_failed}/{len(results)} runs failed")


if __name__ == "__main__":
    main()
