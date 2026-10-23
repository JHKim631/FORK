#!/usr/bin/env bash
# Serve the training grader (configs/grader/qwen36_35b.yaml) on the GPUs the
# trainer does not use: REPLICAS independent vLLM OpenAI-compatible servers,
# each on TP GPUs, on consecutive ports from BASE_PORT. The grader client
# round-robins across the replicas, so there is no load balancer to run.
#
# Default geometry = an 8-GPU node: the trainer on GPUs 0-3 (scripts/train.sh),
# four TP-1 replicas on GPUs 4-7:
#
#     bash scripts/serve_grader.sh                          # GPUs 4-7, ports 8001-8004
#     REPLICAS=2 GPU_OFFSET=2 bash scripts/serve_grader.sh  # a 4-GPU node: trainer 0-1, grader 2-3
#
# VLLM_BIN points at the grader environment's vllm (requirements/grader.txt):
# the model needs a newer vLLM than the trainer's. The first launch downloads
# the model (~35 GB); point HF_HOME at a persistent volume to pay that once.
# Leave this running; Ctrl-C stops every replica.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
REPLICAS="${REPLICAS:-4}"
TP="${TP:-1}"
BASE_PORT="${BASE_PORT:-8001}"
GPU_OFFSET="${GPU_OFFSET:-4}"          # the first grader GPU
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"   # the chunked-prefill budget per tick
VLLM_BIN="${VLLM_BIN:-vllm}"
# Cudagraph capture must cover max_num_seqs or decode is CPU-bound at high
# concurrency. Space-free JSON on purpose: the value is word-split below.
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---compilation-config={\"max_cudagraph_capture_size\":256}}"
LOG_DIR="${GRADER_LOG_DIR:-outputs/grader}"

need=$(( REPLICAS * TP ))
have=$(nvidia-smi -L 2>/dev/null | wc -l)
if (( GPU_OFFSET + need > have )); then
  echo "ERROR: need GPUs [$GPU_OFFSET..$(( GPU_OFFSET + need - 1 ))] (REPLICAS=$REPLICAS x TP=$TP) but only $have visible." >&2
  exit 1
fi
mkdir -p "$LOG_DIR"

urls=()
pids=()
for (( r=0; r<REPLICAS; r++ )); do
  port=$(( BASE_PORT + r ))
  start=$(( GPU_OFFSET + r * TP ))                      # this replica's contiguous GPU block
  gpus=$(seq -s, "$start" $(( start + TP - 1 )))
  log="$LOG_DIR/replica_${port}.log"
  echo ">> replica $r: GPUs [$gpus] -> http://localhost:${port}/v1   (log: $log)"
  CUDA_VISIBLE_DEVICES="$gpus" "$VLLM_BIN" serve "$MODEL" \
    --port "$port" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    ${EXTRA_VLLM_ARGS} \
    > "$log" 2>&1 &
  pids+=($!)
  urls+=("http://localhost:${port}/v1")
done

# Kill every replica if this script is interrupted or exits.
trap 'echo; echo ">> stopping replicas"; kill "${pids[@]}" 2>/dev/null || true' INT TERM EXIT

echo ">> waiting for the replicas to load (the first launch downloads the model; 10-40 min)..."
for (( r=0; r<REPLICAS; r++ )); do
  port=$(( BASE_PORT + r ))
  printf ">> :%s /health " "$port"
  healthy=0
  for _ in $(seq 1 360); do                             # ~60 min ceiling, for a cold download
    if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
      echo "OK"; healthy=1; break
    fi
    if ! kill -0 "${pids[$r]}" 2>/dev/null; then
      echo "FAILED (process exited early)"; echo "---- tail $LOG_DIR/replica_${port}.log ----" >&2
      tail -n 40 "$LOG_DIR/replica_${port}.log" >&2
      exit 1
    fi
    printf "."; sleep 10
  done
  if (( ! healthy )); then
    echo "FAILED (no /health after 60 min; process still running)"
    echo "---- tail $LOG_DIR/replica_${port}.log ----" >&2
    tail -n 40 "$LOG_DIR/replica_${port}.log" >&2
    exit 1
  fi
done

joined=$(IFS=,; echo "${urls[*]}")
echo
echo "================================================================"
echo "All $REPLICAS replica(s) healthy:  GRADER_URLS=${joined}"
echo "Leave this running; Ctrl-C stops all replicas."
echo "================================================================"
wait
