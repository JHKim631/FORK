#!/usr/bin/env bash
# The training and evaluation data.
#
#   bash scripts/prepare_data.sh             # convert the shipped splits to verl's format (seconds, no network)
#   bash scripts/prepare_data.sh --rebuild   # rebuild the shipped splits and OOD pools from their sources first
#
# The repository ships the prepared splits — data/medical, data/science (the
# curated RubricHub-Science slice) — and the two OOD pools, data/healthbench
# and data/researchqa, with their provenance in each meta.json (data/README.md).
# --rebuild downloads the sources at their pinned revisions and rebuilds every
# file in place; `git status` then shows whether anything changed (it should not).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == "--rebuild" ]]; then
  echo ">> [rebuild] RubricHub-Medical -> data/medical"
  python3 -m fork.data.prepare --config configs/data/medical.yaml
  echo ">> [rebuild] RubricHub-Science -> data/science_raw (the full slice), then the curated data/science"
  python3 -m fork.data.prepare --config configs/data/science_raw.yaml
  python3 -m fork.data.curate --config configs/data/science.yaml
  echo ">> [rebuild] the OOD pools"
  python3 -m fork.data.ood --bench healthbench
  python3 -m fork.data.ood --bench researchqa
fi

for dom in medical science; do
  for split in train eval; do
    [[ -f data/$dom/$split.parquet ]] || { echo "ERROR: data/$dom/$split.parquet missing (git checkout, or --rebuild)" >&2; exit 1; }
    python3 -m fork.data.to_verl --data data/$dom/$split.parquet --out data/$dom/verl/$split.parquet
  done
done
echo ">> done: data/{medical,science}/verl/{train,eval}.parquet"
