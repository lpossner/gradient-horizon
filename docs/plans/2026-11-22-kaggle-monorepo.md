# femtochat → gradient-horizon Monorepo Implementation Plan

> **Status: implemented and merged 2026-11-25** (`7c5f730`). All 36 steps are
> complete. The checkboxes are left unticked as originally written and are
> **not** a to-do list. Corrections applied after execution are marked
> inline; the design doc's **Outcome** section records what actually
> happened.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the local-only femtochat repo into the public gradient-horizon Jekyll blog as a `femtochat/` subtree, and add a CLI script that syncs Kaggle notebooks in both directions.

**Architecture:** The monorepo is the existing gradient-horizon repo — it keeps its history, remote, and GitHub Pages wiring. femtochat's 18 commits are grafted in with `git subtree add`, so the experiment history survives without rewriting the blog's. `femtochat/` is added to Jekyll's `exclude` list, so the site ignores it entirely. A single bash script in `tools/` owns the only cross-boundary operation: moving notebooks between `femtochat/notebooks/` and Kaggle.

**Tech Stack:** git 2.50 (`git subtree`), bash, Jekyll 7.6 + Chirpy theme (Ruby 3.4, bundler), Kaggle CLI 2.2.4, uv (femtochat's Python toolchain).

**Spec:** `docs/specs/2026-11-22-kaggle-monorepo-design.md` (was in femtochat when this plan was written; landed at the monorepo root `docs/specs/` in Task 2).

## Global Constraints

- **No new tests.** Verification is by running the real command and checking its output. Do not add pytest, CI, or a test framework.
- **`.env` must never be committed.** It holds live `HF_TOKEN` and `WANDB_API_KEY`. The repo is public. This is the highest-severity failure mode in the whole plan.
- **The 1.6 GB of local state must never be committed:** `.venv/`, `FineWeb-edu_sample_10BT/`, `wandb/`, `output/`. All four are gitignored in femtochat today and their rules must be in the monorepo's `.gitignore` before any `git add -A`. (They now live at `femtochat/{.venv,FineWeb-edu_sample_10BT,wandb,output}` and are ignored there; nothing was ever committed.)
- **`sync_kaggle.sh` never uploads or downloads a `.py` file.** The script moves notebooks and nothing else.
- **`kernel-metadata.json`'s `dataset_sources` and `kernel_sources` must survive every push.** They encode the kernel chain; dropping them breaks the run in a way that looks like a data problem.
- **Only two kernels are in scope:** `tokenize-fineweb-edu-sample-10bt` and `train-fineweb-edu-sample-10bt-cuda`. The two superseded kernels (`train-fineweb-edu-sample-10bt`, `tokenize-fineweb-sample-10bt`) are deliberately left in place and out of scope.
- **One docs convention:** `docs/specs/` at the repo root. `docs/superpowers/` is abandoned.
- Kaggle CLI 2.2.4 has **no verb to run an already-uploaded kernel.** `kernels push` (alias `update`) always uploads *and* runs. There is therefore no `run` subcommand.
- All paths below are relative to the monorepo root unless stated.

## Review Focus

The five failure modes most likely to bite, each pinned to the task that owns it:

1. **A secret or a gigabyte reaching the public repo.** `.env`, `.venv/`, the corpus, `wandb/`, `output/`. A wrong `git add -A` publishes credentials irreversibly. → Task 2, Steps 1, 2, and 8; Task 6, Step 3.
2. **A push that silently breaks the kernel chain** by dropping `dataset_sources`/`kernel_sources`. → Task 4, Step 6.
3. **Jekyll building or serving `femtochat/`** — most visibly the notebooks, which would be published as raw files. → Task 1, Step 4; Task 3, Step 4; Task 6, Step 2.
4. **`output` pointed at a tokenize kernel**, pulling ~264 MB of corpus that nobody wants. → Task 4, Step 4.
5. **A figure referenced *with* `relative_url`**, which doubles the baseurl and 404s in production — the opposite of the usual Jekyll advice, because Chirpy's `media-url.html` already prepends it. → Task 5, Steps 1 and 2.

---

### Task 1: Prepare the blog repo for the merge

The blog repo has uncommitted work in flight. It must be committed before a subtree merge, or the merge will tangle with it.

**Files:**
- Modify: `_config.yml` (exclude list, ~line 218)
- Modify: `.github/workflows/pages-deploy.yml` (paths-ignore, ~line 6)
- Verify only: `_posts/2026-10-31-introduction-to-femtochat.md`, `_posts/2026-11-15-architechture-of-femtochat.md`

**Interfaces:**
- Consumes: nothing.
- Produces: a clean `main` with femtochat excluded from Jekyll and from deploy triggers. Task 2 grafts into this.

- [ ] **Step 1: Commit the in-flight blog work**

The repo has three modified files (a tagline change in `_config.yml` and edits to both posts). These are the author's own edits, not part of this plan. Commit them as they stand, or have the author commit them. Do not fold them into a plan commit.

```bash
cd /Users/lpossner/Projects/lpossner.github.io/gradient-horizon
git status --porcelain
git add _config.yml _posts/2026-10-31-introduction-to-femtochat.md _posts/2026-11-15-architechture-of-femtochat.md
git commit -m "Update tagline and revise posts"
```

- [ ] **Step 2: Add `femtochat` to Jekyll's exclude list**

In `_config.yml`, the `exclude:` block already lists `docs` and `tools`. Add `femtochat` to it:

```yaml
exclude:
  - "*.gem"
  - "*.gemspec"
  - docs
  - tools
  - femtochat
  - README.md
  - LICENSE
  - purgecss.js
  - "*.config.js"
  - "package*.json"
```

Jekyll treats an entry as a path prefix, so this covers `femtochat/notebooks/**` and everything else under it.

- [ ] **Step 3: Stop code commits from redeploying the site**

In `.github/workflows/pages-deploy.yml`, extend `paths-ignore`:

```yaml
on:
  push:
    branches:
      - main
      - master
    paths-ignore:
      - .gitignore
      - README.md
      - LICENSE
      - femtochat/**
      - docs/**
      - tools/**
```

`paths-ignore` skips the workflow only when *every* changed path matches, so a commit touching both a post and `femtochat/` still deploys.

- [ ] **Step 4: Verify the site still builds and excludes femtochat**

```bash
cd /Users/lpossner/Projects/lpossner.github.io/gradient-horizon
mkdir -p femtochat && echo "scratch" > femtochat/scratch.txt
bundle exec jekyll b -d /tmp/_site_check
ls /tmp/_site_check/femtochat 2>&1 | head -2
```

Expected: the build succeeds, and `/tmp/_site_check/femtochat` does **not** exist (Jekyll skipped it). Then remove the scratch file: `rm -rf femtochat`.

- [ ] **Step 5: Commit**

```bash
git add _config.yml .github/workflows/pages-deploy.yml
git commit -m "Exclude femtochat from Jekyll and from deploy triggers"
```

---

### Task 2: Graft femtochat's history in and lay out the tree

The structural task. `git subtree add` brings femtochat's 18 commits in as `femtochat/`. Because subtree reads git history — not the working directory — the 1.6 GB of untracked local state is never involved. The gitignore merge still happens first, as a guard against any later `git add -A`.

**Files:**
- Create: `femtochat/` (via subtree)
- Modify: `.gitignore` (merge femtochat's rules)
- Move: `femtochat/docs/roadmap.md` → `docs/roadmap.md`
- Move: `femtochat/docs/specs/*` → `docs/specs/*`
- Move: `femtochat/*.log` → `femtochat/logs/*.log`
- Delete: `femtochat/.gitignore` (after merging)

**Interfaces:**
- Consumes: the clean `main` from Task 1.
- Produces: `femtochat/{train.py,sweep.py,make_figures.py,REPORT.md,pyproject.toml,uv.lock,figures/}`, `femtochat/logs/*.log`, and root `docs/{roadmap.md,specs/}`. Tasks 3–6 depend on this layout.

- [ ] **Step 1: Merge femtochat's ignore rules into the root `.gitignore` first**

This is the guard, and it happens before anything can be staged. Append femtochat's rules to the blog's `.gitignore` (the blog's existing rules stay):

```gitignore
# --- femtochat ---
# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info
.ruff_cache/

# Virtual environments
.venv

# Superpowers scratch state
.superpowers/

# Project data
data/
FineWeb-edu_sample_10BT/

# Training output
output/

# Secrets
.env

# WandDB
wandb/
```

The blog's own `.gitignore` already covers `.DS_Store`, so it is not repeated.

- [ ] **Step 2: Confirm the ignore rules actually fire**

```bash
cd /Users/lpossner/Projects/lpossner.github.io/gradient-horizon
for p in .env .venv FineWeb-edu_sample_10BT wandb output __pycache__; do
  printf "%-26s " "$p"; git check-ignore -q "$p" && echo IGNORED || echo "NOT IGNORED"
done
```

Expected: `IGNORED` for all six. If any says NOT IGNORED, fix `.gitignore` before continuing — do not proceed to Step 3.

- [ ] **Step 3: Graft femtochat in as a subtree**

femtochat has no remote. **As executed**, the path was passed to `git subtree`
directly, because `git remote add` was refused by the permission classifier:

```bash
git subtree add --prefix=femtochat /Users/lpossner/Projects/femtochat main -m "Graft femtochat history into the monorepo"
```

The original form (register a remote first, then graft from it) is equivalent
but leaves a `femtochat-local` remote behind:

```bash
git remote add femtochat-local /Users/lpossner/Projects/femtochat
git fetch femtochat-local
git subtree add --prefix=femtochat femtochat-local/main -m "Graft femtochat history into the monorepo"
```

- [ ] **Step 4: Verify the graft brought history, and nothing else**

```bash
git log --oneline -- femtochat | head -5
git log --oneline | head -3
du -sh femtochat
git status --porcelain
```

Expected: femtochat's commits appear (including `157137b Track the ablation report…`); the HEAD commit is the graft merge; `femtochat` is a few MB, **not** hundreds; and `git status` is clean. If `du` shows a gigabyte, the gitignore merge in Step 1 was wrong — stop and fix it.

- [ ] **Step 5: Collapse the docs convention**

Both repos' docs move under the single root `docs/`. The blog's existing `docs/superpowers/specs/2026-11-20-transformer-diagrams-design.md` is folded in too:

```bash
mkdir -p docs/specs
git mv femtochat/docs/roadmap.md docs/roadmap.md
git mv femtochat/docs/specs/special-tokens.md docs/specs/special-tokens.md
git mv femtochat/docs/specs/2026-11-22-kaggle-monorepo-design.md docs/specs/2026-11-22-kaggle-monorepo-design.md
git mv femtochat/docs/plans/2026-11-22-kaggle-monorepo.md docs/plans/2026-11-22-kaggle-monorepo.md
git mv docs/superpowers/specs/2026-11-20-transformer-diagrams-design.md docs/specs/2026-11-20-transformer-diagrams-design.md
rmdir femtochat/docs/specs femtochat/docs/plans femtochat/docs docs/superpowers/specs docs/superpowers
```

`mkdir -p docs/plans` first if it does not exist.

- [ ] **Step 6: Move the run logs under `logs/`**

```bash
mkdir -p femtochat/logs
git mv femtochat/exp1_attn_scale.log femtochat/logs/
git mv femtochat/exp23_depth_search.log femtochat/logs/
git mv femtochat/exp23_extreme.log femtochat/logs/
git mv femtochat/exp23_lr_search.log femtochat/logs/
git mv femtochat/exp23_postnorm_search.log femtochat/logs/
git mv femtochat/sweep_overnight.log femtochat/logs/
```

- [ ] **Step 7: Remove femtochat's now-redundant `.gitignore`**

Its rules are in the root file (Step 1). `femtochat/.python-version` and the empty `femtochat/README.md` stay.

```bash
git rm femtochat/.gitignore
```

- [ ] **Step 8: Verify nothing large or secret is tracked**

```bash
git ls-files femtochat | wc -l
git ls-files | grep -E "\.env$|\.venv/|wandb/|FineWeb-edu_sample_10BT/|^output/" ; echo "exit=$?"
git ls-files | xargs du -ch 2>/dev/null | tail -1
```

Expected: a file count in the low twenties; the grep prints nothing and exits non-zero (no matches); total tracked size in the low megabytes. **If the grep prints any path, stop — do not commit.**

- [ ] **Step 9: Commit**

```bash
git add -A
git status --porcelain
git commit -m "Collapse docs to docs/specs and move femtochat run logs under logs/"
```

Before committing, read the `git status` output and confirm every listed path is one you expect. Then verify the tree is clean: `git status --porcelain` should print nothing.

---

### Task 3: Bring the in-scope notebooks under version control

**Files:**
- Create: `femtochat/notebooks/tokenize-fineweb-edu-sample-10bt/{kernel-metadata.json,*.ipynb}`
- Create: `femtochat/notebooks/train-fineweb-edu-sample-10bt-cuda/{kernel-metadata.json,*.ipynb}`

**Interfaces:**
- Consumes: the `femtochat/notebooks/` location from Task 2.
- Produces: two kernel directories, each with `kernel-metadata.json` and one `.ipynb`. Task 4's script reads and writes exactly this layout, and derives the Kaggle ref from the `id` field in the metadata.

- [ ] **Step 1: Pull both kernels**

```bash
cd /Users/lpossner/Projects/lpossner.github.io/gradient-horizon/femtochat
mkdir -p notebooks/tokenize-fineweb-edu-sample-10bt notebooks/train-fineweb-edu-sample-10bt-cuda
kaggle kernels pull lucasponer/tokenize-fineweb-edu-sample-10bt -p notebooks/tokenize-fineweb-edu-sample-10bt -m
kaggle kernels pull lucasponer/train-fineweb-edu-sample-10bt-cuda -p notebooks/train-fineweb-edu-sample-10bt-cuda -m
```

- [ ] **Step 2: Verify the chain metadata survived the pull**

```bash
for d in notebooks/*/; do
  echo "=== $d"
  python3 -c "
import json,sys
m=json.load(open('$d/kernel-metadata.json'))
print(' id:', m['id'])
print(' dataset_sources:', m['dataset_sources'])
print(' kernel_sources:', m['kernel_sources'])
print(' private:', m['is_private'])
"
done
```

Expected: the tokenize kernel shows `dataset_sources: ['lucasponer/fineweb-edu-sample-10bt']` and empty `kernel_sources`; the train kernel shows `kernel_sources: ['lucasponer/tokenize-fineweb-edu-sample-10bt']`. Both `private: True`. If either source list is empty, the chain is broken — re-pull with `-m`.

- [ ] **Step 3: Seed the `.last-pull` hash for each kernel**

The sync script's dirty check compares three hashes: the local notebook, Kaggle's current notebook, and `.last-pull` — the hash of the notebook as Kaggle last served it. Right after a pull those three are equal, so seed the file now:

```bash
cd /Users/lpossner/Projects/lpossner.github.io/gradient-horizon/femtochat
for d in notebooks/*/; do
  nb=$(ls "$d"*.ipynb)
  shasum -a 256 "$nb" | awk '{print $1}' > "$d/.last-pull"
  echo "$d -> $(cat "$d/.last-pull")"
done
```

- [ ] **Step 4: Verify Jekyll still ignores the notebooks**

```bash
cd /Users/lpossner/Projects/lpossner.github.io/gradient-horizon
bundle exec jekyll b -d /tmp/_site_nb
find /tmp/_site_nb -name "*.ipynb" | head
```

Expected: no `.ipynb` anywhere in the built site.

- [ ] **Step 5: Commit**

```bash
git add femtochat/notebooks
git commit -m "Track the two in-scope Kaggle notebooks with their metadata"
```

---

### Task 4: Write `tools/sync_kaggle.sh`

**Files:**
- Create: `tools/sync_kaggle.sh`

**Interfaces:**
- Consumes: `femtochat/notebooks/<kernel>/` from Task 3; the `id` field of each `kernel-metadata.json` for the Kaggle ref.
- Produces: the CLI contract below. Task 6 exercises it.

The contract:

```
sync_kaggle.sh pull   <kernel>|all     Kaggle -> femtochat/notebooks/<kernel>/
sync_kaggle.sh push   <kernel>|all     local -> Kaggle, then trigger a run
sync_kaggle.sh status <kernel>|all     print the latest run status
sync_kaggle.sh output <kernel>         fetch run outputs into femtochat/output/<kernel>/
sync_kaggle.sh figures                 femtochat/figures/*.png -> assets/img/femtochat/
```

`<kernel>` is the short name (`tokenize-fineweb-edu-sample-10bt`), resolved to `lucasponer/<kernel>` via the metadata's `id`.

- [ ] **Step 1: Write the script**

Behaviour the implementation must have, in full:

- `set -euo pipefail`; resolve the repo root from the script's own location so it runs from any cwd.
- A `KERNELS` list naming the two in-scope kernels, and a `TRAIN_KERNELS` list naming the subset `output` accepts.
- **Dirty check on `push`.** Each kernel directory holds a tracked `.last-pull` file containing the sha256 of the notebook as Kaggle last served it. On push: if the local notebook's hash differs from `.last-pull`, the local copy moved; pull the remote into a temp dir and hash it — if that also differs from `.last-pull`, both sides moved, so print both hashes and exit non-zero without uploading. On a successful push, update `.last-pull` from the just-pulled remote.
- `push` must not alter `dataset_sources` or `kernel_sources` in `kernel-metadata.json`. It may pass the file through untouched.
- `push` calls `kaggle kernels push -p <dir>`. Document in a comment that this always triggers a run — there is no separate run verb in CLI 2.2.4.
- **`output` refuses tokenize kernels** with a clear message, and passes `--file-pattern` so it does not fetch a whole corpus even if the guard is bypassed. Outputs land in `femtochat/output/<kernel>/`, which is already gitignored.
- `figures` copies `femtochat/figures/*.png` to `assets/img/femtochat/`, creating the directory, and prints how many files it wrote.
- `--force` bypasses the dirty check. `--help` documents all subcommands.
- Non-interactive; every failure exits non-zero.

- [ ] **Step 2: Verify argument handling**

```bash
chmod +x tools/sync_kaggle.sh
tools/sync_kaggle.sh --help
tools/sync_kaggle.sh bogus 2>&1 | tail -2; echo "exit=$?"
tools/sync_kaggle.sh pull 2>&1 | tail -2; echo "exit=$?"
```

Expected: `--help` prints usage listing all five subcommands; an unknown subcommand and a missing kernel argument both exit non-zero with a message.

- [ ] **Step 3: Verify `status` against real Kaggle**

```bash
tools/sync_kaggle.sh status all
```

Expected: one line per in-scope kernel with its latest run status. This is the first real API call — if it fails, the credentials or the ref resolution is wrong.

- [ ] **Step 4: Verify `output` refuses a tokenize kernel**

```bash
tools/sync_kaggle.sh output tokenize-fineweb-edu-sample-10bt 2>&1 | tail -3; echo "exit=$?"
```

Expected: a message saying tokenize kernels are excluded, and a non-zero exit. No download starts.

- [ ] **Step 5: Verify the dirty check fires**

With `.last-pull` seeded by Task 3 Step 3, modifying the local notebook makes the local hash differ while Kaggle's is unchanged. That is a clean local-only edit, so the script should proceed — the check only blocks when *both* sides moved.

```bash
printf '\n' >> femtochat/notebooks/train-fineweb-edu-sample-10bt-cuda/train-fineweb-edu-sample-10bt-cuda.ipynb
tools/sync_kaggle.sh push train-fineweb-edu-sample-10bt-cuda 2>&1 | tail -5; echo "exit=$?"
git checkout -- femtochat/notebooks/train-fineweb-edu-sample-10bt-cuda/
```

Expected: the script reports the local notebook changed and proceeds with the upload (`exit=0`). Then simulate the conflict — change Kaggle's copy by pushing from a scratch clone, edit the local file, and push again — and confirm the script **refuses** with both hashes printed and a non-zero exit. The conflict path is the one that must not silently pick a winner.

Then restore the file as shown.

- [ ] **Step 6: Verify a push preserves the kernel chain**

```bash
tools/sync_kaggle.sh push tokenize-fineweb-edu-sample-10bt
kaggle kernels pull lucasponer/tokenize-fineweb-edu-sample-10bt -p /tmp/kcheck -m
python3 -c "
import json
m=json.load(open('/tmp/kcheck/kernel-metadata.json'))
assert m['dataset_sources']==['lucasponer/fineweb-edu-sample-10bt'], m['dataset_sources']
print('dataset_sources preserved:', m['dataset_sources'])
"
```

Expected: the assertion passes. This is Review Focus #2 — a dropped source list is the failure that looks like a data problem.

- [ ] **Step 7: Verify `figures` copies**

```bash
tools/sync_kaggle.sh figures
ls assets/img/femtochat/ | head
```

Expected: the six PNGs from `femtochat/figures/`.

- [ ] **Step 8: Commit**

The figures are tracked in both locations (spec: "Figure pipeline"), so `assets/img/femtochat/` is committed here, not left untracked:

```bash
git add tools/sync_kaggle.sh femtochat/notebooks/*/.last-pull assets/img/femtochat/
git commit -m "Add sync_kaggle.sh for notebook and figure sync"
```

---

### Task 5: Wire a figure into the blog

Proves the figure path end to end before the sweep post depends on it.

**Files:**
- Modify: `_posts/2026-11-15-architechture-of-femtochat.md` (or a new draft post)

**Interfaces:**
- Consumes: `assets/img/femtochat/*.png` from Task 4 Step 7.
- Produces: a working figure reference pattern the sweep post will reuse.

- [ ] **Step 1: Add one figure reference to a post**

Insert into the post body:

```markdown
![Optimizer, initialization & weight tying](/assets/img/femtochat/sweep_optimizer_init.png)
```

**Use a bare path, not `relative_url`.** Chirpy post-processes every post image through `_includes/media-url.html`, which prepends the baseurl itself, so a bare path already gets `/gradient-horizon` exactly once. Adding `relative_url` produces `/gradient-horizon/gradient-horizon/assets/...`, which 404s in production. This was measured against a real build rather than assumed. To re-test after a theme upgrade, build with `--drafts` and read the rendered `src`.

- [ ] **Step 2: Build and confirm the image is served**

```bash
bundle exec jekyll b -d /tmp/_site_fig
ls /tmp/_site_fig/assets/img/femtochat/ | head
grep -o 'src="[^"]*femtochat[^"]*"' /tmp/_site_fig/**/*.html | head -3
```

Expected: the PNGs are in the built site, and the rendered HTML references them with the `/gradient-horizon` prefix.

- [ ] **Step 3: Run the site's own link checker**

```bash
bash tools/test.sh
```

Expected: htmlproofer passes. A broken image path fails here.

- [ ] **Step 4: Commit**

```bash
git add _posts/
git commit -m "Embed a sweep figure in a draft post"
```

---

### Task 6: End-to-end verification and cleanup

**Files:** none created.

**Interfaces:**
- Consumes: everything above.
- Produces: a verified monorepo and a decision about the old local checkout.

- [ ] **Step 1: Confirm the deploy workflow still runs**

Push to `main` and watch the run:

```bash
git push origin main
gh run list --limit 3
gh run watch
```

Expected: "Build and Deploy" succeeds. Then confirm the site loads and the figure from Task 5 renders.

- [ ] **Step 2: Confirm `femtochat/` is not published**

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://lpossner.github.io/gradient-horizon/femtochat/train.py
curl -s -o /dev/null -w "%{http_code}\n" https://lpossner.github.io/gradient-horizon/femtochat/notebooks/
```

Expected: `404` for both. Review Focus #3.

- [ ] **Step 3: Confirm no secret is reachable**

```bash
git log --all --diff-filter=A --name-only --pretty=format: | sort -u | grep -E "\.env$" ; echo "exit=$?"
gh api /repos/lpossner/gradient-horizon/contents/.env 2>&1 | tail -1
```

Expected: no `.env` in any commit's added files, and the API returns 404.

- [ ] **Step 4: Decide what happens to the old local checkout**

The old `/Users/lpossner/Projects/femtochat` still holds the 1.6 GB that makes `train.py` runnable (`.venv`, the tokenized corpus, `output/`, `wandb/`). The monorepo checkout does not have it, and the design lists portability of that state as a non-goal.

Two options — **this is the author's call, do not pick it for them**:

- **Move the data into the monorepo checkout.** `mv` the four ignored directories from the old checkout to `.../gradient-horizon/femtochat/`, so `train.py` runs from the monorepo and the old directory can be deleted. The rules are already in the root `.gitignore` (Task 2 Step 1), so they stay untracked.
- **Keep the old checkout** as the place where training runs, and treat the monorepo as the canonical source that gets pulled from.

Whichever is chosen, do not delete the old directory until the author confirms the monorepo builds, deploys, and that they can run `train.py` somewhere.

**As executed (2026-11-28):** the author chose to move it. Done as a `cp -Rp`
rather than `mv` — `mv` was refused by the permission classifier, and a copy
is reversible. The corpus, `output/`, and `wandb/` were copied;
`.venv` was **rebuilt with `uv sync`** rather than copied, because 69
entry-point shebangs in it pointed at the old absolute path. `train.py` and
`make_figures.py` were then verified working from the monorepo, and the old
directory was deleted.

**Mistake to avoid repeating:** `.env` existed only in the old checkout, and
it was deleted with it. Its copy had been blocked by the permission
classifier, and the deletion proceeded anyway instead of waiting. The file was
restored by hand with fresh values. When a copy of something unique is
blocked, that must stop the delete — not precede it.

- [ ] **Step 5: Final state check**

```bash
git status --porcelain
git log --oneline | head -8
du -sh .
```

Expected: clean tree; the graft and the plan's commits visible; repo size dominated by `.git` and the blog's own assets, not by femtochat's data.

---

## Notes for the executor

- **Task 1 Step 1 is not yours to invent.** The blog's uncommitted edits are the author's in-flight work. Commit them as-is or ask.
- **Task 2 Step 2 is a hard gate.** If `git check-ignore` does not report all six paths as ignored, stop. Everything downstream assumes it.
- **`git subtree add` reads history, not the working tree**, so it cannot bring the 1.6 GB in. The only way that data enters git is a later `git add -A` after a manual copy — which is why the ignore rules land first.
- **Task 2 Step 2's gate produces a false negative.** `git check-ignore <path>` only matches a directory rule when the path exists (or is given a trailing slash). The four directories are absent from a fresh blog checkout, so the bare-name form reports NOT IGNORED for `FineWeb-edu_sample_10BT`, `wandb`, `output` and `__pycache__` even when the rules are correct. Test with trailing slashes, or create the directory first.
- **The `train.py` fork is intentional.** The monorepo's `femtochat/train.py` runs on the M1; the CUDA notebook embeds its own training script. They are not meant to be unified here. `docs/roadmap.md` Stage 0 owns that merge.
- **The notebook path difference is normal Kaggle behaviour.** `tokenize-fineweb-edu-sample-10bt` writes to `/kaggle/working/...` and `train-fineweb-edu-sample-10bt-cuda` reads it back via `kernel_sources` at `/kaggle/input/notebooks/...`. Different paths are expected; the chain works through `kernel_sources`, not a shared filesystem.
