#!/usr/bin/env bash
# License: BSD 3-Clause
#
# Before/after parity check: run compare-spectrograms + compare-model in the pixi
# env of a base git ref and of the current working tree, on identical segments,
# then plot the two side by side with plot_parity.py. Each side runs in its own
# `compare` env, or in its `test` env if it has no `compare` env (sox_tensorflow
# 0.1.3, whose test env has TensorFlow).
#
# usage: comparison/parity_check.sh [-b BASE_REF] [-n N_SAMPLES] [-o OUT_DIR] [-f FIGURE] [AUDIO...]
#
#   -b  git ref to compare against            [default: main]
#   -n  segments per audio file               [default: n_samples in config.yaml]
#   -o  output directory                      [default: tmp/parity]
#   -f  figure path                           [default: <OUT_DIR>/parity.png]
#   AUDIO  files, directories or S3 URIs      [default: audio in config.yaml]
#
# Requires the sox binary; falls back to conda-forge sox via `pixi exec` if not on PATH.
# The base checkout and its pixi env are cached in <OUT_DIR>/.base-<sha> for reruns.
set -euo pipefail

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
BASE_REF=main
N_SAMPLES=
OUT=$ROOT/tmp/parity
FIGURE=
while getopts "b:n:o:f:h" opt; do
  case $opt in
    b) BASE_REF=$OPTARG ;;
    n) N_SAMPLES=$OPTARG ;;
    o) mkdir -p "$OPTARG"; OUT=$(cd "$OPTARG" && pwd) ;;
    f) FIGURE=$OPTARG ;;
    *) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  esac
done
shift $((OPTIND - 1))
FIGURE=${FIGURE:-$OUT/parity.png}

# absolute paths for local audio, since each tool runs from its own checkout
AUDIO=()
for a in "$@"; do
  if [[ $a == s3://* ]]; then AUDIO+=("$a"); else AUDIO+=("$(cd "$(dirname "$a")" && pwd)/$(basename "$a")"); fi
done

if ! command -v sox >/dev/null; then
  PATH=$(dirname "$(pixi exec -s sox which sox)"):$PATH
fi
export TF_USE_LEGACY_KERAS=1

BASE_SHA=$(git -C "$ROOT" rev-parse --short "$BASE_REF")
BASE_DIR=$OUT/.base-$BASE_SHA
if [[ ! -d $BASE_DIR ]]; then
  mkdir -p "$BASE_DIR"
  git -C "$ROOT" archive "$BASE_SHA" | tar -x -C "$BASE_DIR"
fi

get_env() {  # $1=project dir: its compare env if it has one, else test
  if grep -qE '^compare *=' "$1/pyproject.toml"; then echo compare; else echo test; fi
}

run_side() {  # $1=side (base|head)  $2=label  $3=project dir  $4=pixi env
  local label=$2 proj=$3 env=$4 dest=$OUT/$1
  rm -rf "$dest" && mkdir -p "$dest"
  echo "== $label: $proj (pixi env $env)"
  (
    cd "$proj"
    pixi install -e "$env" --frozen
    pixi run -e "$env" --frozen python -c "import json, sys, numpy, tensorflow; \
json.dump({'label': sys.argv[1], 'numpy': numpy.__version__, 'tensorflow': tensorflow.__version__}, \
open(sys.argv[2], 'w'))" "$label" "$dest/versions.json"
    pixi run -e "$env" --frozen compare-spectrograms ${AUDIO[@]+"${AUDIO[@]}"} \
      -o "$dest" ${N_SAMPLES:+-n "$N_SAMPLES"}
    pixi run -e "$env" --frozen compare-model -s "$dest/spectrograms" -o "$dest"
  )
}

run_side base "$BASE_REF" "$BASE_DIR" "$(get_env "$BASE_DIR")"
run_side head "$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)" "$ROOT" "$(get_env "$ROOT")"

cd "$ROOT"
pixi run -e compare --frozen python comparison/plot_parity.py "$OUT/base" "$OUT/head" --figure "$FIGURE"
