#!/usr/bin/env bash
#
# sync_kaggle.sh -- move Kaggle notebooks and blog figures between this repo
# and Kaggle.
#
# Ownership rule: .py files are owned locally and never travel through this
# script. Notebooks are owned by Kaggle's web editor. So this script moves
# notebooks and nothing else, plus a mechanical copy of generated figures.
#
# Usage:
#   sync_kaggle.sh pull   <kernel>|all   Kaggle -> femtochat/notebooks/<kernel>/
#   sync_kaggle.sh push   <kernel>|all   local -> Kaggle  (this also RUNS it)
#   sync_kaggle.sh status <kernel>|all   latest run status
#   sync_kaggle.sh output <kernel>       run outputs -> femtochat/output/<kernel>/
#   sync_kaggle.sh figures               femtochat/figures/*.png -> assets/img/femtochat/
#
# Options:
#   -f, --force    push: skip the both-sides-changed check
#       --dry-run  push: do every check, but do not upload or run
#   -h, --help     this message
#
# NOTE ON RUNNING: Kaggle CLI 2.2.4 has no verb that runs an
# already-uploaded kernel. `kaggle kernels push` (alias `update`) uploads AND
# runs, and its only options are --path, --timeout and --accelerator. There
# is therefore deliberately no `run` subcommand: pushing is the only way to
# trigger a run, and every push starts one.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTEBOOKS_DIR="$REPO_ROOT/femtochat/notebooks"
# Fetched kernel outputs land here, one directory per kernel.
OUTPUT_DIR="$REPO_ROOT/femtochat/output"
# Per-post folders are named <YYYY-MM-DD|placeholder>-<post>. Figures live
# beside the code that produced them, under that post's output/figures/.
# Nothing here is post-specific: the first one that has plots wins, so a
# second post needs no change to this script.
FIGURES_SRC="$(ls -d "$REPO_ROOT"/femtochat/*/output/figures 2>/dev/null | head -n 1)"
FIGURES_DST="$REPO_ROOT/assets/img/femtochat"

# Kernels this script manages. The two superseded kernels
# (train-fineweb-edu-sample-10bt, tokenize-fineweb-sample-10bt) are
# deliberately out of scope and are left in place on Kaggle.
KERNELS=(
  tokenize-fineweb-edu-sample-10bt
  train-fineweb-edu-sample-10bt-cuda
)

# `output` accepts only these. The tokenize kernel's output is the ~264 MB
# tokenized corpus -- an input to the train kernel, not an artefact anyone
# downloads -- so it is refused outright.
TRAIN_KERNELS=(
  train-fineweb-edu-sample-10bt-cuda
)

# Even for a train kernel, only fetch plausible artefacts. This is a second
# guard behind the TRAIN_KERNELS check above: without it a stray .bin.gz in
# the output directory would be pulled down in full.
OUTPUT_FILE_PATTERN='.*\.(pt|pth|json|txt|log|csv)$'

usage() {
  sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

err() { echo "error: $*" >&2; }

# The Kaggle CLI prints a version-outdated notice, and it reaches both
# stdout and stderr, so it cannot be dropped by redirecting one stream.
# Filter it by content instead, and preserve the CLI's exit status.
# NOTE: `out="$(cmd)"` as a bare assignment trips `set -e` on failure and
# aborts the function before the captured text can be printed, so the
# assignment is guarded with `|| status=$?` and the output always shown.
kaggle_quiet() {
  local out status=0
  out="$(kaggle "$@" 2>&1)" || status=$?
  printf '%s\n' "$out" | grep -v 'outdated `kaggle` version' || true
  return "$status"
}

kernel_metadata() {
  local kernel="$1" m="$NOTEBOOKS_DIR/$kernel/kernel-metadata.json"
  if [[ ! -f "$m" ]]; then
    err "no metadata at $m"
    return 1
  fi
  python3 - "$m" <<'PY' || { err "could not read $m"; return 1; }
import json, sys
try:
    print(json.load(open(sys.argv[1]))["id"])
except Exception as e:
    sys.exit(f"metadata is not valid JSON with an 'id': {e}")
PY
}

kernel_ref() { kernel_metadata "$1"; }

# The file `push` actually uploads is the one kernel-metadata.json names in
# `code_file` -- not "whatever .ipynb happens to be in the directory". Hashing
# a different file than the one uploaded would guard the wrong bytes.
notebook_path() {
  local kernel="$1" nb
  nb="$(python3 - "$NOTEBOOKS_DIR/$kernel/kernel-metadata.json" <<'PY' || true
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("code_file", ""))
except Exception:
    pass
PY
)"
  if [[ -z "$nb" ]]; then
    err "kernel-metadata.json for $kernel has no 'code_file'"
    return 1
  fi
  if [[ ! -f "$NOTEBOOKS_DIR/$kernel/$nb" ]]; then
    err "code_file '$nb' named by $kernel/kernel-metadata.json does not exist"
    return 1
  fi
  printf '%s\n' "$NOTEBOOKS_DIR/$kernel/$nb"
}

# Hash a notebook by its parsed structure, not its bytes. The Kaggle CLI
# re-serialises a notebook on pull (json.dumps with default separators),
# while the copies on disk use compact separators, so a byte hash of a
# pushed-then-pulled notebook differs from the file that was pushed. That
# difference would show up as a phantom "both sides changed". Canonicalising
# through json removes the difference without hiding a real edit.
file_hash() {
  python3 - "$1" <<'PY'
import hashlib, json, sys
path = sys.argv[1]
try:
    with open(path, "rb") as fh:
        nb = json.load(fh)
except json.JSONDecodeError as e:
    sys.exit(f"{path} is not valid notebook JSON: {e}")
except OSError as e:
    sys.exit(f"cannot read {path}: {e}")
blob = json.dumps(nb, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(blob).hexdigest())
PY
}

# Hash of the notebook as Kaggle currently serves it. Pulls into a temp dir
# and discards it -- nothing is written into notebooks/ except by `pull`.
remote_notebook_hash() {
  local kernel="$1" ref tmp nb h
  ref="$(kernel_ref "$kernel")" || return 1
  tmp="$(mktemp -d)"
  if ! kaggle kernels pull "$ref" -p "$tmp" >/dev/null 2>&1; then
    rm -rf "$tmp"
    return 1
  fi
  nb="$(find "$tmp" -maxdepth 1 -name '*.ipynb' | head -1)"
  if [[ -z "$nb" ]]; then
    rm -rf "$tmp"
    return 1
  fi
  h="$(file_hash "$nb")"
  rm -rf "$tmp"
  printf '%s\n' "$h"
}

require_kernel() {
  local kernel="$1" k
  for k in "${KERNELS[@]}"; do
    [[ "$k" == "$kernel" ]] && return 0
  done
  err "unknown kernel: $kernel"
  err "known kernels: ${KERNELS[*]}"
  return 1
}

is_train_kernel() {
  local kernel="$1" k
  for k in "${TRAIN_KERNELS[@]}"; do
    [[ "$k" == "$kernel" ]] && return 0
  done
  return 1
}

cmd_pull() {
  local kernel="$1" ref nb base_file local_hash
  ref="$(kernel_ref "$kernel")" || return 1
  nb="$(notebook_path "$kernel")" || return 1
  base_file="$NOTEBOOKS_DIR/$kernel/.last-pull"

  # Pull overwrites the local notebook, so it needs the same protection push
  # gives Kaggle's copy: if the local file has edits that were never pushed,
  # those edits are about to be destroyed. Refuse instead.
  if [[ "$FORCE" != "1" && -f "$base_file" ]]; then
    local_hash="$(file_hash "$nb")"
    if [[ "$local_hash" != "$(cat "$base_file")" ]]; then
      err "the local notebook for $kernel has changes that were never pushed"
      echo "  local: $local_hash" >&2
      echo "  base:  $(cat "$base_file")  (from $base_file)" >&2
      echo "  Pulling would discard them. Push first, or re-run with --force." >&2
      return 1
    fi
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dry run: would pull $ref into femtochat/notebooks/$kernel"
    return 0
  fi

  echo "pulling $ref -> femtochat/notebooks/$kernel"
  kaggle_quiet kernels pull "$ref" -p "$NOTEBOOKS_DIR/$kernel" -m
  # The freshly pulled notebook is now the agreed state.
  file_hash "$nb" > "$base_file"
}

cmd_push() {
  local kernel="$1" ref nb base_file local_hash base remote_hash
  ref="$(kernel_ref "$kernel")" || return 1
  nb="$(notebook_path "$kernel")" || return 1
  base_file="$NOTEBOOKS_DIR/$kernel/.last-pull"
  local_hash="$(file_hash "$nb")"

  if [[ "$FORCE" != "1" ]]; then
    # No recorded baseline means we cannot tell whether Kaggle's copy moved.
    # A missing baseline must refuse, not silently skip the check: skipping
    # is how an edit made in Kaggle's editor gets overwritten.
    if [[ ! -f "$base_file" ]]; then
      err "no $base_file -- cannot tell whether Kaggle's copy has changed"
      echo "  Run 'sync_kaggle.sh pull $kernel' once to establish a baseline," >&2
      echo "  or re-run with --force to overwrite Kaggle's copy unconditionally." >&2
      return 1
    fi
    base="$(cat "$base_file")"

    if [[ "$local_hash" != "$base" ]]; then
      # The local notebook moved. Did Kaggle's move too? If it did, one of
      # the two edits would be lost, so refuse and show both. A failure to
      # *verify* is not evidence of "no conflict": if the remote hash cannot
      # be read, refuse rather than push blind.
      if ! remote_hash="$(remote_notebook_hash "$kernel")"; then
        err "could not read Kaggle's copy of $kernel, so the conflict check cannot run"
        echo "  Refusing to push: a transient network or auth failure must not" >&2
        echo "  be read as 'no conflict'. Retry, or use --force to overwrite." >&2
        return 1
      fi
      if [[ "$remote_hash" != "$base" ]]; then
        err "both sides changed -- refusing to push $kernel"
        echo "  local:  $local_hash" >&2
        echo "  remote: $remote_hash" >&2
        echo "  base:   $base  (from $base_file)" >&2
        echo "  Pull first, merge by hand, or re-run with --force." >&2
        return 1
      fi
    else
      # Local is untouched. If Kaggle's copy moved, pushing would overwrite
      # an edit made in Kaggle's editor. A failed fetch refuses here too.
      if ! remote_hash="$(remote_notebook_hash "$kernel")"; then
        err "could not read Kaggle's copy of $kernel, so the conflict check cannot run"
        echo "  Refusing to push. Retry, or use --force to overwrite." >&2
        return 1
      fi
      if [[ "$remote_hash" != "$base" ]]; then
        err "Kaggle's copy of $kernel has changed since the last pull"
        echo "  Run 'sync_kaggle.sh pull $kernel' to bring it home first." >&2
        return 1
      fi
    fi
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dry run: would push $ref from femtochat/notebooks/$kernel (and start a run)"
    return 0
  fi

  echo "pushing $ref from femtochat/notebooks/$kernel (this also starts a run)"
  kaggle kernels push -p "$NOTEBOOKS_DIR/$kernel"

  # Kaggle may normalise the notebook on ingest, so record the agreed state
  # from what Kaggle now serves rather than from the local file. If that
  # read fails, fall back to the local hash rather than leaving the stale
  # baseline that would make the next push report a phantom conflict.
  if remote_hash="$(remote_notebook_hash "$kernel")"; then
    printf '%s\n' "$remote_hash" > "$base_file"
  else
    printf '%s\n' "$local_hash" > "$base_file"
  fi
}

cmd_status() {
  local kernel="$1" ref
  ref="$(kernel_ref "$kernel")" || return 1
  printf '%-40s ' "$kernel"
  kaggle_quiet kernels status "$ref"
}

cmd_output() {
  local kernel="$1" ref
  if ! is_train_kernel "$kernel"; then
    err "$kernel is not a train kernel -- refusing to download its output"
    err "its output is the tokenized corpus, which belongs in /kaggle/input, not here"
    err "train kernels: ${TRAIN_KERNELS[*]}"
    return 1
  fi
  ref="$(kernel_ref "$kernel")"
  mkdir -p "$OUTPUT_DIR/$kernel"
  echo "fetching outputs of $ref -> femtochat/output/$kernel"
  kaggle kernels output "$ref" -p "$OUTPUT_DIR/$kernel" \
    --file-pattern "$OUTPUT_FILE_PATTERN"
}

cmd_figures() {
  local f n=0
  mkdir -p "$FIGURES_DST"
  shopt -s nullglob
  for f in "$FIGURES_SRC"/*.png; do
    cp "$f" "$FIGURES_DST/"
    n=$((n + 1))
  done
  shopt -u nullglob
  echo "copied $n figure(s) to assets/img/femtochat"
}

SUBCOMMAND=""
KERNEL=""
FORCE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    -f | --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -*)
      err "unknown option: $1"
      usage >&2
      exit 1
      ;;
    *)
      if [[ -z "$SUBCOMMAND" ]]; then
        SUBCOMMAND="$1"
      elif [[ -z "$KERNEL" ]]; then
        KERNEL="$1"
      else
        err "unexpected argument: $1"
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$SUBCOMMAND" ]]; then
  usage >&2
  exit 1
fi

# Every subcommand except `figures` takes a kernel, and `all` fans out.
if [[ "$SUBCOMMAND" == "figures" ]]; then
  # `figures` takes no kernel; a stray argument is a mistake, not a no-op.
  if [[ -n "$KERNEL" ]]; then
    err "'figures' takes no arguments (got: $KERNEL)"
    exit 1
  fi
  cmd_figures
  exit 0
fi

# --force and --dry-run are meaningful only for the notebook subcommands.
# Silently ignoring them elsewhere would let `--dry-run pull` do a real pull.
if [[ "$SUBCOMMAND" == "status" || "$SUBCOMMAND" == "output" ]]; then
  if [[ "$FORCE" == "1" || "$DRY_RUN" == "1" ]]; then
    err "--force and --dry-run apply to pull/push only, not to $SUBCOMMAND"
    exit 1
  fi
fi

if [[ -z "$KERNEL" ]]; then
  err "$SUBCOMMAND needs a kernel argument (or 'all')"
  err "known kernels: ${KERNELS[*]}"
  exit 1
fi

targets=()
if [[ "$KERNEL" == "all" ]]; then
  if [[ "$SUBCOMMAND" == "output" ]]; then
    err "'output' takes one kernel, not 'all'"
    exit 1
  fi
  targets=("${KERNELS[@]}")
else
  require_kernel "$KERNEL" || exit 1
  targets=("$KERNEL")
fi

failed=0
for k in "${targets[@]}"; do
  case "$SUBCOMMAND" in
    pull) cmd_pull "$k" || failed=1 ;;
    push) cmd_push "$k" || failed=1 ;;
    status) cmd_status "$k" || failed=1 ;;
    output) cmd_output "$k" || failed=1 ;;
    *)
      err "unknown subcommand: $SUBCOMMAND"
      usage >&2
      exit 1
      ;;
  esac
done

# `all` runs every kernel and reports failure if any of them failed.
exit "$failed"
