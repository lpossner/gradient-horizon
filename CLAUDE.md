# gradient-horizon

One repo, two halves: a Jekyll/Chirpy blog (the site) and `femtochat/` (the code
and results it writes about). `femtochat/` is excluded from Jekyll; nothing in
it is published except the figures copied into `assets/`.

## Layout

```
_posts/                     published posts
_drafts/                    unpublished posts
assets/img/femtochat/       plots the blog serves — a COPY, not the source
docs/                       specs, plans, roadmap
tools/sync_kaggle.sh        Kaggle notebook sync + figure copy
tools/run.sh, test.sh       Jekyll serve / build + htmlproofer
femtochat/
  input/                    FineWeb corpus (gitignored)
  _archive/notebooks/       the old Kaggle kernels
  <post-name>/              one folder per blog post
  xxxx-xx-xx-<post-name>/   ...with xxxx-xx-xx until the post exists
```

Per-post folders hold the code, results and figures that post discusses:
`train.py`, `sweep.py`, `make_figures.py`, `REPORT.md`, `logs/`,
`output/tensorboard/` (gitignored), `output/figures/` (tracked).

## Figures are generated, never hand-edited

The chain is **tensorboard events → `make_figures.py` → `<post>/output/figures/`
→ `assets/img/femtochat/` → published site.** Only the third arrow needs running
by hand; the deploy workflow does it before every build.

- Never edit a PNG in `assets/img/femtochat/`. Edit the plotting code and
  regenerate. A hand-touched plot is overwritten on the next build.
- `make_figures.py` must run with the project venv, from any directory:
  `femtochat/.venv/bin/python femtochat/<post>/make_figures.py`
  If packages are missing: `cd femtochat && uv sync` (Python 3.11, pinned in
  `.python-version`; `uv.lock` is authoritative).
- After regenerating, run `bash tools/sync_kaggle.sh figures` if working on the
  site locally, and commit the PNGs. CI copies again, so a forgotten sync never
  publishes a stale plot — but it can leave your local diff stale.

## Paths

Every script resolves `input/` and `output/` from **its own location**, not the
cwd, so all of them work from any directory. Do not reintroduce `Path.cwd()` or
bare relative paths. `train.py` honours `FEMTOCHAT_INPUT_DIR` and
`FEMTOCHAT_OUTPUT_DIR`; that is how the Kaggle notebook points at
`/kaggle/input` and `/kaggle/working` without editing the code.

`sync_kaggle.sh` finds figures by globbing `femtochat/*/output/figures` — the
first match wins, so it is not post-aware. A second post with figures needs that
changed before its plots can be copied.

## Kaggle

`tools/sync_kaggle.sh pull|push|status|output` operate on
`femtochat/_archive/notebooks/`. Notebooks are owned by Kaggle's web editor;
`.py` files are owned locally and never travel through the script. `push` also
*starts a run* — Kaggle CLI has no verb to run an already-uploaded kernel.

## Site

- `bash tools/run.sh` — local serve at 127.0.0.1:4000
- `bash tools/test.sh` — production build + htmlproofer
- Deploy: GitHub Actions on push to `main`. The workflow's `paths:` allowlist
  controls triggering; `femtochat/**/output/figures/**` is deliberately
  re-included so a regenerated plot deploys.
- Post permalinks are `/posts/:title/`, derived from the filename. Renaming a
  post file changes its public URL and anything linking to it 404s.

## Conventions

- No tests, no CI beyond deploy. Verification is running the real command.
- `femtochat/.env` holds live `WANDB_API_KEY` / `HF_TOKEN`. Never commit it; the
  repo is public.
- Gitignored and must stay that way: `femtochat/input/`, any `output/` contents
  except `output/figures/*.png`, `wandb/`, `.venv/`, `.env`, `.playwright-mcp/`.
  The `.gitignore` pattern for output is anchored under `femtochat/` on purpose
  — see the comment there before editing it.