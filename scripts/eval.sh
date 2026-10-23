#!/usr/bin/env bash
# Evaluate one model on its domain — the paper's Tables 1 and 2 for one row.
#
#   bash scripts/eval.sh base_medical                      # the untrained policy (Qwen2.5-7B-Instruct)
#   bash scripts/eval.sh fork_beta1.25_medical             # a trained run, its last checkpoint
#   bash scripts/eval.sh fork_beta1.25_medical --step 250  # a specific checkpoint
#   bash scripts/eval.sh standard_science --judge local    # the training grader as judge (needs the replicas)
#
# Steps, each guarded by its output file (rerun the same command to resume):
#   1 merge      the FSDP checkpoint -> an HF model dir under outputs/hf/<run>/step_<n> (not for base)
#   2 generate   the grader subset (100 prompts) and the OOD pool, on GPU $GEN_GPU
#   3 rubric_based   recall and precision under the judge             -> rubric_based.<judge>/
#   4 rubric_free    pairwise vs the comparator under prompts p1 and p2 -> rubric_free.<judge>.<p>/
#   5 ood            the benchmark's own grading of the OOD responses  -> ood.<judge>/
#                    and the pairwise judging of the OOD responses vs the comparator (p1)
#   6 report         the row in the paper's columns (python -m fork.eval.report)
#
# The comparator (--vs, default base_<domain>) must have been evaluated first,
# with the same judge: its responses are the other side of every pair.
# --judge gpt54 (default) needs OPENAI_API_KEY in .env; --judge local needs
# the grader replicas (scripts/serve_grader.sh; GRADER_URLS or the eval
# config's local_judge.base_urls).
#
# Env: GEN_GPU (0), GRADER_URLS, TP (1: vLLM tensor parallel for generation), FORK_EVAL_DIR (outputs/eval).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
[[ -f .env ]] && set -a && source .env && set +a

TARGET="${1:?usage: eval.sh <run_id | base_<domain>> [--step N] [--judge gpt54|local] [--vs <label>]}"; shift
STEP=""; JUDGE="gpt54"; VS=""
while (( $# )); do
  case "$1" in
    --step) STEP="$2"; shift 2;;
    --judge) JUDGE="$2"; shift 2;;
    --vs) VS="$2"; shift 2;;
    *) echo "unknown argument $1" >&2; exit 1;;
  esac
done
EVAL_DIR="${FORK_EVAL_DIR:-outputs/eval}"
GEN_GPU="${GEN_GPU:-0}"
TP="${TP:-1}"

if [[ "$TARGET" == base_medical || "$TARGET" == base_science ]]; then
  DOMAIN="${TARGET#base_}"; LABEL="$TARGET"
  MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/grpo.yaml'))['policy'])")
else
  RUN_CFG="configs/train/${TARGET}.yaml"
  [[ -f "$RUN_CFG" ]] || { echo "ERROR: $RUN_CFG missing — <run_id> names a config under configs/train/" >&2; exit 1; }
  eval "$(python3 -m fork.launch env "$RUN_CFG")"      # FORK_DOMAIN, RUN_ID, ...
  DOMAIN="$FORK_DOMAIN"
  CKPTS="outputs/train/${TARGET}/ckpts"
  if [[ -z "$STEP" ]]; then
    STEP=$(ls -d "$CKPTS"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
    [[ -n "$STEP" ]] || { echo "ERROR: no checkpoint under $CKPTS" >&2; exit 1; }
  fi
  LABEL="${TARGET}@${STEP}"
  MODEL="outputs/hf/${TARGET}/step_${STEP}"
  if [[ ! -f "$MODEL/config.json" ]]; then
    echo ">> [1] merge $CKPTS/global_step_$STEP -> $MODEL"
    [[ -d "$CKPTS/global_step_$STEP/actor" ]] || { echo "ERROR: $CKPTS/global_step_$STEP/actor missing" >&2; exit 1; }
    python3 -m verl.model_merger merge --backend fsdp --local_dir "$CKPTS/global_step_$STEP/actor" --target_dir "$MODEL.partial"
    mv "$MODEL.partial" "$MODEL"
  fi
fi
CFG="configs/eval/${DOMAIN}.yaml"
VS="${VS:-base_${DOMAIN}}"
OOD_POOL=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['ood']['pool'])")
D="$EVAL_DIR/$LABEL"
mkdir -p "$D"
JUDGE_ARGS=(--judge "$JUDGE")
if [[ "$JUDGE" == local ]]; then
  [[ -n "${GRADER_URLS:-}" ]] && JUDGE_ARGS+=(--base-urls "$GRADER_URLS")
fi
echo ">> eval $LABEL | domain $DOMAIN | model $MODEL | judge $JUDGE | vs $VS"

echo ">> [2] generate"
[[ -f "$D/grader_responses.jsonl" ]] || CUDA_VISIBLE_DEVICES="$GEN_GPU" python3 -m fork.eval.generate \
    --config "$CFG" --model "$MODEL" --label "$LABEL" --subset grader --out-dir "$EVAL_DIR" --tensor-parallel-size "$TP"
[[ -f "$D/ood_responses.jsonl" ]] || CUDA_VISIBLE_DEVICES="$GEN_GPU" python3 -m fork.eval.generate \
    --config "$CFG" --model "$MODEL" --label "$LABEL" --pool "$OOD_POOL" --out-dir "$EVAL_DIR" --tensor-parallel-size "$TP"

echo ">> [3] rubric-based: recall + precision"
python3 -m fork.eval.rubric_based --config "$CFG" --responses "$D/grader_responses.jsonl" --label "$LABEL" \
    --out-dir "$D/rubric_based.$JUDGE" "${JUDGE_ARGS[@]}"

if [[ "$LABEL" != "$VS" ]]; then
  V="$EVAL_DIR/$VS"
  [[ -f "$V/grader_responses.jsonl" ]] || { echo "ERROR: the comparator $VS has no grader_responses.jsonl — evaluate it first (bash scripts/eval.sh $VS)" >&2; exit 1; }
  echo ">> [4] rubric-free: pairwise vs $VS (p1, p2)"
  for P in p1 p2; do
    OUT="$D/rubric_free.$JUDGE.$P/vs_$VS.jsonl"
    [[ -f "$OUT" ]] || python3 -m fork.eval.rubric_free --config "$CFG" --prompt "$P" \
        --x-label "$LABEL" --x "$D/grader_responses.jsonl" --y-label "$VS" --y "$V/grader_responses.jsonl" \
        --out-file "$OUT" "${JUDGE_ARGS[@]}"
  done
fi

echo ">> [5] OOD: the benchmark's own grading, then pairwise vs $VS (p1)"
[[ -f "$D/ood.$JUDGE/graded.jsonl" ]] || python3 -m fork.eval.ood --config "$CFG" \
    --responses "$D/ood_responses.jsonl" --label "$LABEL" --out-dir "$D/ood.$JUDGE" "${JUDGE_ARGS[@]}"
if [[ "$LABEL" != "$VS" && -f "$EVAL_DIR/$VS/ood_responses.jsonl" ]]; then
  OUT="$D/ood_rubric_free.$JUDGE.p1/vs_$VS.jsonl"
  [[ -f "$OUT" ]] || python3 -m fork.eval.rubric_free --config "$CFG" --prompt p1 \
      --x-label "$LABEL" --x "$D/ood_responses.jsonl" --y-label "$VS" --y "$EVAL_DIR/$VS/ood_responses.jsonl" \
      --out-file "$OUT" "${JUDGE_ARGS[@]}"
fi

echo ">> [6] report"
python3 -m fork.eval.report --eval-dir "$EVAL_DIR" --label "$LABEL" --vs "$VS" --judge "$JUDGE"
