#!/usr/bin/env bash
# The reward trajectory of one run (paper Figure 1): every checkpoint scored
# on the 300-prompt trajectory subset with the TRAINING grader as judge —
# rubric recall, precision and the win rate against the base policy.
#
#   bash scripts/eval.sh base_medical --judge local        # first: the base side, once per domain
#   bash scripts/trajectory.sh fork_beta1.25_medical       # then every checkpoint of the run
#
# Needs the grader replicas (scripts/serve_grader.sh; GRADER_URLS) and a GPU
# for generation (GEN_GPU). Each checkpoint is merged to outputs/hf/<run>/step_<n>,
# generated on the trajectory subset into outputs/eval/<run>@<n>/, scored by
# fork.eval.rubric_based under the local judge and judged pairwise (p1)
# against base_<domain>'s trajectory responses. Every step is guarded by its
# output file; rerun to resume. The table at the end is
# `python -m fork.eval.report --trajectory <run> --vs base_<domain> --judge local`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
[[ -f .env ]] && set -a && source .env && set +a

RUN="${1:?usage: trajectory.sh <run_id>}"
EVAL_DIR="${FORK_EVAL_DIR:-outputs/eval}"
GEN_GPU="${GEN_GPU:-0}"
TP="${TP:-1}"
RUN_CFG="configs/train/${RUN}.yaml"
[[ -f "$RUN_CFG" ]] || { echo "ERROR: $RUN_CFG missing" >&2; exit 1; }
eval "$(python3 -m fork.launch env "$RUN_CFG")"
DOMAIN="$FORK_DOMAIN"; CFG="configs/eval/${DOMAIN}.yaml"; VS="base_${DOMAIN}"
JUDGE_ARGS=(--judge local); [[ -n "${GRADER_URLS:-}" ]] && JUDGE_ARGS+=(--base-urls "$GRADER_URLS")
B="$EVAL_DIR/$VS"
if [[ ! -f "$B/trajectory_responses.jsonl" ]]; then
  echo ">> the base side: generating $VS on the trajectory subset"
  MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/grpo.yaml'))['policy'])")
  CUDA_VISIBLE_DEVICES="$GEN_GPU" python3 -m fork.eval.generate --config "$CFG" --model "$MODEL" --label "$VS" \
      --subset trajectory --out-dir "$EVAL_DIR" --tensor-parallel-size "$TP"
fi
[[ -f "$B/rubric_based.local/recall_scored.jsonl" && -f "$B/rubric_based.local/precision_scored.jsonl" ]] || \
  python3 -m fork.eval.rubric_based --config "$CFG" --responses "$B/trajectory_responses.jsonl" --label "$VS" \
      --out-dir "$B/rubric_based.local" "${JUDGE_ARGS[@]}"

CKPTS="outputs/train/${RUN}/ckpts"
for STEP in $(ls -d "$CKPTS"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n); do
  LABEL="${RUN}@${STEP}"; MODEL="outputs/hf/${RUN}/step_${STEP}"; D="$EVAL_DIR/$LABEL"
  echo ">> checkpoint $STEP"
  if [[ ! -f "$MODEL/config.json" ]]; then
    python3 -m verl.model_merger merge --backend fsdp --local_dir "$CKPTS/global_step_$STEP/actor" --target_dir "$MODEL.partial"
    mv "$MODEL.partial" "$MODEL"
  fi
  [[ -f "$D/trajectory_responses.jsonl" ]] || CUDA_VISIBLE_DEVICES="$GEN_GPU" python3 -m fork.eval.generate \
      --config "$CFG" --model "$MODEL" --label "$LABEL" --subset trajectory --out-dir "$EVAL_DIR" --tensor-parallel-size "$TP"
  python3 -m fork.eval.rubric_based --config "$CFG" --responses "$D/trajectory_responses.jsonl" --label "$LABEL" \
      --out-dir "$D/rubric_based.local" "${JUDGE_ARGS[@]}"
  OUT="$D/rubric_free.local.p1/vs_$VS.jsonl"
  [[ -f "$OUT" ]] || python3 -m fork.eval.rubric_free --config "$CFG" --prompt p1 \
      --x-label "$LABEL" --x "$D/trajectory_responses.jsonl" --y-label "$VS" --y "$B/trajectory_responses.jsonl" \
      --out-file "$OUT" "${JUDGE_ARGS[@]}"
done
python3 -m fork.eval.report --eval-dir "$EVAL_DIR" --trajectory "$RUN" --vs "$VS" --judge local
