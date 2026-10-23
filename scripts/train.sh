#!/usr/bin/env bash
# verl GRPO training — one run per invocation, resumable.
#
#   bash scripts/train.sh configs/train/fork_beta1.25_medical.yaml
#   bash scripts/train.sh configs/train/standard_medical.yaml --print   # show the verl command only
#
# Prerequisites: the data built (scripts/prepare_data.sh), the grader serving
# (scripts/serve_grader.sh), the trainer environment active (requirements/trainer.txt),
# and the keys in .env (WANDB_API_KEY is optional: without it the logger is the console).
#
# Default geometry = an 8-GPU node: the trainer on GPUs 0-3, the grader's four
# replicas on 4-7 (GRADER_URLS below). verl requires the 2,048 rollouts of a
# step to divide across the trainer GPUs, so TRAIN_GPUS must name 1, 2, 4 or 8.
# A 4-GPU node: TRAIN_GPUS=0,1 and the grader on 2-3 (REPLICAS=2 GPU_OFFSET=2
# for serve_grader.sh), with GRADER_URLS listing its two replicas.
#
# RESUMABLE: rerun the same command after a stop. Training resumes from the
# last checkpoint in <run dir>/ckpts; the reward's step logs past that
# checkpoint are moved aside (never deleted) so the audit trail stays aligned.
#
# Knobs (env): TRAIN_GPUS (0,1,2,3), GRADER_URLS (the four replicas' URLs),
#   MICRO_BSZ (8: sequences per GPU per forward/backward), LOGP_MICRO_BSZ (16),
#   ROLLOUT_GPU_MEM (0.5), OPTIM_OFFLOAD (False; True on GPUs that cannot hold
#   the optimizer beside the rollout engine), SAVE_FREQ (configs/grpo.yaml's),
#   TOTAL_STEPS (empty = the configured epochs), LOGGER, WANDB_PROJECT (fork),
#   CKPT_DIR (an absolute checkpoint root instead of <run dir>/ckpts),
#   FORK_ABORT_FRAC (0.05: the grader-failure fraction that aborts a step).
set -euo pipefail

RUN_CFG="${1:?usage: train.sh configs/train/<run>.yaml [--print]}"
PRINT_ONLY=0; [[ "${2:-}" == "--print" ]] && PRINT_ONLY=1
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
[[ -f .env ]] && set -a && source .env && set +a

# The run's switches and data paths (fork.launch validates the yaml):
# RUN_ID FORK_REWARD FORK_BETA FORK_PRECISION FORK_DOMAIN EPOCHS TRAIN_FILES VAL_FILES POLICY
env_lines=$(python3 -m fork.launch env "$RUN_CFG") || { echo "ERROR: invalid run config $RUN_CFG" >&2; exit 1; }
eval "$env_lines"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
N_GPUS=$(awk -F, '{print NF}' <<<"$TRAIN_GPUS")
GRADER_URLS="${GRADER_URLS:-http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1,http://localhost:8004/v1}"
GRADER_CONFIG="${GRADER_CONFIG:-configs/grader/qwen36_35b.yaml}"
MICRO_BSZ="${MICRO_BSZ:-8}"
LOGP_MICRO_BSZ="${LOGP_MICRO_BSZ:-16}"
ROLLOUT_GPU_MEM="${ROLLOUT_GPU_MEM:-0.5}"
OPTIM_OFFLOAD="${OPTIM_OFFLOAD:-False}"
TOTAL_STEPS="${TOTAL_STEPS:-}"
WANDB_PROJECT="${WANDB_PROJECT:-fork}"
if [[ -z "${LOGGER:-}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then LOGGER='["console","wandb"]'; else LOGGER='["console"]'; fi
fi
export WANDB_PROJECT

run_dir="outputs/train/${RUN_ID}"
if [[ -n "${CKPT_DIR:-}" && "$CKPT_DIR" != /* ]]; then
  echo "ERROR: CKPT_DIR must be an absolute path (got '$CKPT_DIR')" >&2; exit 1
fi
ckpt_dir="${CKPT_DIR:-$(pwd)/$run_dir/ckpts}"

args_lines=$(python3 -m fork.launch args "$RUN_CFG" \
    --n-gpus "$N_GPUS" --micro-bsz "$MICRO_BSZ" --logp-micro-bsz "$LOGP_MICRO_BSZ" \
    --rollout-gpu-mem "$ROLLOUT_GPU_MEM" --optim-offload "$OPTIM_OFFLOAD" \
    --logger "$LOGGER" --project "$WANDB_PROJECT" --ckpt-dir "$ckpt_dir" --run-dir "$run_dir" \
    --reward-path "$(pwd)/src/fork/rewards/verl_reward.py" ${TOTAL_STEPS:+--total-steps "$TOTAL_STEPS"}) \
  || { echo "ERROR: could not render the verl command for $RUN_CFG" >&2; exit 1; }
mapfile -t ARGS <<<"$args_lines"

if (( PRINT_ONLY )); then
  echo "run=$RUN_ID reward=$FORK_REWARD${FORK_BETA:+ beta=$FORK_BETA}${FORK_PRECISION:+ precision=$FORK_PRECISION} domain=$FORK_DOMAIN epochs=$EPOCHS gpus=$TRAIN_GPUS grader=$GRADER_URLS"
  printf 'python3 -m verl.trainer.main_ppo \\\n'; printf '    %s \\\n' "${ARGS[@]}"; echo
  exit 0
fi

[[ -f "$TRAIN_FILES" && -f "$VAL_FILES" ]] || { echo "ERROR: $TRAIN_FILES / $VAL_FILES missing — run scripts/prepare_data.sh first" >&2; exit 1; }
mkdir -p "$run_dir"
[[ -f "$GRADER_CONFIG" ]] || { echo "ERROR: $GRADER_CONFIG missing" >&2; exit 1; }
for u in ${GRADER_URLS//,/ }; do
  curl -sfm 5 "${u%/v1}/health" >/dev/null 2>&1 || { echo "ERROR: grader replica $u is not healthy — start scripts/serve_grader.sh first" >&2; exit 1; }
done
rm -f "$run_dir/ABORT"   # clear the sticky grader-failure abort of a previous launch (verl_reward.py)

# Resume bookkeeping: verl restarts at the last checkpoint, so the reward's
# step logs written after it are stale. Move them aside (kept, never deleted)
# so the audit index continues from the checkpoint.
ckpt_marker="$ckpt_dir/latest_checkpointed_iteration.txt"
if [[ -f "$ckpt_marker" ]] || compgen -G "$run_dir/grader_io/step_*.jsonl" >/dev/null 2>&1; then
  ckpt_step=0; [[ -f "$ckpt_marker" ]] && ckpt_step=$(tr -dc '0-9' < "$ckpt_marker")
  python3 - "$run_dir" "${ckpt_step:-0}" <<'PY' || echo "WARN: stale step-log trim failed (continuing)" >&2
import json, os, shutil, sys, time
from pathlib import Path
run_dir, ckpt = Path(sys.argv[1]), int(sys.argv[2])
stale = run_dir / "resume_stale" / time.strftime("%Y%m%d_%H%M%S")
moved = 0
d = run_dir / "grader_io"
if d.is_dir():
    for p in sorted(d.glob("step_*.jsonl")):
        idx = p.stem.removeprefix("step_")
        if idx.isdigit() and int(idx) >= ckpt:          # audit index i <-> global step i + 1
            (stale / "grader_io").mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(stale / "grader_io" / p.name)); moved += 1
rl = run_dir / "reward_log.jsonl"
if rl.exists():
    keep, drop = [], []
    for line in rl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            st = int(json.loads(line).get("step", -1))
        except Exception:
            st = -1
        (drop if st >= ckpt else keep).append(line)
    if drop:
        stale.mkdir(parents=True, exist_ok=True)
        (stale / "reward_log.tail.jsonl").write_text("\n".join(drop) + "\n", encoding="utf-8")
        tmp = rl.with_name("reward_log.jsonl.tmp")
        tmp.write_text(("\n".join(keep) + "\n") if keep else "", encoding="utf-8")
        os.replace(tmp, rl); moved += len(drop)
print(f">> resume at checkpoint {ckpt}: {moved} stale step records moved to {stale}" if moved
      else f">> resume at checkpoint {ckpt}: the step logs already end there")
PY
fi

# The FORK_* environment reaches the reward function only because verl starts
# its own Ray cluster inside this process (workers inherit this shell). A
# pre-existing cluster would hand the reward function a stale environment.
if [[ -n "${RAY_ADDRESS:-}" ]]; then
  echo "ERROR: RAY_ADDRESS='$RAY_ADDRESS' is set — verl would attach to that cluster and the FORK_* reward env would not propagate. unset it (and 'ray stop') first." >&2
  exit 1
fi
if command -v ray >/dev/null 2>&1 && timeout 15 ray status >/dev/null 2>&1; then
  echo "ERROR: a Ray cluster is already running on this node — 'ray stop' it first; verl must start its own cluster so the FORK_* reward env propagates." >&2
  exit 1
fi

# Reward wiring: src/fork/rewards/verl_reward.py reads these in the trainer's
# driver process (verl's custom_reward_function config carries no user kwargs).
export FORK_REWARD FORK_BETA FORK_PRECISION FORK_DOMAIN
export FORK_GRADER_CONFIG="$GRADER_CONFIG"
export FORK_GRADER_URLS="$GRADER_URLS"
export FORK_RUN_DIR="$run_dir"
export FORK_ABORT_FRAC="${FORK_ABORT_FRAC:-0.05}"
export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"

# Provenance: a launch record appended to <run dir>/manifest.json.
python3 - "$RUN_CFG" "$run_dir" "$GRADER_CONFIG" "$ckpt_dir" <<'PY' || echo "WARN: manifest write failed (continuing)" >&2
import hashlib, json, os, subprocess, sys, time
run_cfg, run_dir, grader_cfg, ckpt_dir = sys.argv[1:5]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cmd": f"train.sh {run_cfg}",
         "git_sha": git("rev-parse", "HEAD") or None, "git_dirty": bool(git("status", "--porcelain")),
         "run_cfg": run_cfg, "run_cfg_sha256": sha(run_cfg), "grpo_cfg_sha256": sha("configs/grpo.yaml"),
         "grader_config": grader_cfg, "grader_config_sha256": sha(grader_cfg),
         **{k: os.environ.get(k) or None for k in ("FORK_REWARD", "FORK_BETA", "FORK_PRECISION", "FORK_DOMAIN")},
         "wandb_project": os.environ.get("WANDB_PROJECT"), "resumed_from_ckpt": None}
it = os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt")
if os.path.exists(it):
    entry["resumed_from_ckpt"] = open(it).read().strip()
mpath = os.path.join(run_dir, "manifest.json")
m = json.load(open(mpath)) if os.path.exists(mpath) else {"run": os.path.basename(run_dir)}
m.setdefault("launches", []).append(entry)
json.dump(m, open(mpath, "w"), indent=2)
PY

echo ">> run=$RUN_ID reward=$FORK_REWARD${FORK_BETA:+ beta=$FORK_BETA}${FORK_PRECISION:+ precision=$FORK_PRECISION} domain=$FORK_DOMAIN epochs=$EPOCHS gpus=$TRAIN_GPUS grader=$GRADER_URLS"
python3 -m verl.trainer.main_ppo "${ARGS[@]}" 2>&1 | tee -a "$run_dir/train.log"
