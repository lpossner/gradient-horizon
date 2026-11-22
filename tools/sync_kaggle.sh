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
OUTPUT_DIR="$REPO_ROOT/femtochat/output"
FIGURES_SRC="$REPO_ROOT/femtochat/figures"
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
# Filter it by content instead, and preserve the CLI's exit status (a
# pipeline would otherwise report grep's).
kaggle_quiet() {
  local out status
  out="$(kaggle "$@" 2>&1)"
  status=$?
  printf '%s\n' "$out" | grep -v 'outdated `kaggle` version' || true
  return $status
}

kernel_ref() {
  local kernel="$1"
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' \
    "$NOTEBOOKS_DIR/$kernel/kernel-metadata.json"
}

notebook_path() {
  local kernel="$1" nb
  nb="$(find "$NOTEBOOKS_DIR/$kernel" -maxdepth 1 -name '*.ipynb' | head -1)"
  if [[ -z "$nb" ]]; then
    err "no notebook found in $NOTEBOOKS_DIR/$kernel"
    return 1
  fi
  printf '%s\n' "$nb"
}

file_hash() { shasum -a 256 "$1" | awk '{print $1}'; }

# Hash of the notebook as Kaggle currently serves it. Pulls into a temp dir
# and discards it -- nothing is written into notebooks/ except by `pull`.
remote_notebook_hash() {
  local kernel="$1" ref tmp nb h
  ref="$(kernel_ref "$kernel")"
  tmp="$(mktemp -d)"
  if ! kaggle kernels pull "$ref" -p "$tmp" >/dev/null 2>&1; then
    rm -rf "$tmp"
    return 1
  fi
  nb="$(find "$tmp" -maxdepth 1 -name '*.ipynb' | head -1)"
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
  local kernel="$1" ref
  ref="$(kernel_ref "$kernel")"
  echo "pulling $ref -> femtochat/notebooks/$kernel"
  kaggle_quiet kernels pull "$ref" -p "$NOTEBOOKS_DIR/$kernel" -m
  # The freshly pulled notebook is now the agreed state.
  file_hash "$(notebook_path "$kernel")" > "$NOTEBOOKS_DIR/$kernel/.last-pull"
}

cmd_push() {
  local kernel="$1" ref nb base_file local_hash base remote_hash
  ref="$(kernel_ref "$kernel")"
  nb="$(notebook_path "$kernel")"
  base_file="$NOTEBOOKS_DIR/$kernel/.last-pull"
  local_hash="$(file_hash "$nb")"

  if [[ "$FORCE" != "1" && -f "$base_file" ]]; then
    base="$(cat "$base_file")"
    if [[ "$local_hash" != "$base" ]]; then
      # The local notebook moved. Did Kaggle's move too? If it did, one of
      # the two edits would be lost, so refuse and show both.
      if remote_hash="$(remote_notebook_hash "$kernel")"; then
        if [[ "$remote_hash" != "$base" ]]; then
          err "both sides changed -- refusing to push $kernel"
          echo "  local:  $local_hash" >&2
          echo "  remote: $remote_hash" >&2
          echo "  base:   $base  (from $base_file)" >&2
          echo "  Pull first, merge by hand, or re-run with --force." >&2
          return 1
        fi
      else
        echo "warning: could not fetch the remote notebook; skipping the conflict check" >&2
      fi
    elif [[ "$(remote_notebook_hash "$kernel" 2>/dev/null || true)" != "$base" ]]; then
      # Local is untouched but Kaggle's copy moved: pushing would overwrite
      # an edit made in Kaggle's editor. Never do that silently.
      err "Kaggle's copy of $kernel has changed since the last pull"
      echo "  Run 'sync_kaggle.sh pull $kernel' to bring it home first." >&2
      return 1
    fi
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "dry run: would push $ref from femtochat/notebooks/$kernel (and start a run)"
    return 0
  fi

  echo "pushing $ref from femtochat/notebooks/$kernel (this also starts a run)"
  kaggle kernels push -p "$NOTEBOOKS_DIR/$kernel"

  # Kaggle may normalise the notebook on ingest, so record the agreed state
  # from what Kaggle now serves rather than from the local file.
  if remote_hash="$(remote_notebook_hash "$kernel")"; then
    printf '%s\n' "$remote_hash" > "$base_file"
  fi
}

cmd_status() {
  local kernel="$1"
  printf '%-40s ' "$kernel"
  kaggle_quiet kernels status "$(kernel_ref "$kernel")"
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
  cmd_figures
  exit 0
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
  require_kernel "$KERNEL"
  targets=("$KERNEL")
fi

for k in "${targets[@]}"; do
  case "$SUBCOMMAND" in
    pull) cmd_pull "$k" ;;
    push) cmd_push "$k" ;;
    status) cmd_status "$k" ;;
    output) cmd_output "$k" ;;
    *)
      err "unknown subcommand: $SUBCOMMAND"
      usage >&2
      exit 1
      ;;
  esac
done
