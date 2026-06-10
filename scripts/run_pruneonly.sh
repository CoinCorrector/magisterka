#!/usr/bin/env bash
# run_pruneonly.sh — natywna ewaluacja pruning-only (bez re-save, bez eksportu HF).
#
# Pipeline (zwalidowany 2026-06-10, kontener NeMo nvcr.io/nvidia/nemo:25.09):
#   1. deploy checkpointu NeMo na serwerze OpenAI (PyTriton + FastAPI :8080) z legacy_ckpt=True
#      -> legacy_ckpt=True degraduje brak kluczy _extra_state z fatal error do ostrzeżenia,
#         dzięki czemu SUROWY sprunowany checkpoint ładuje się bez distillacji/re-save.
#   2. lm-eval (backend local-completions) odpytuje endpoint -> zadania loglikelihood (acc_norm)
#      działają, bo endpoint zwraca logprobs+echo. Wyniki logowane do Weights & Biases.
#   3. ubicie serwera.
#
# URUCHAMIAĆ W KONTENERZE NeMo. Wymagania w kontenerze:
#   - nemo.collections.llm.deploy, lm_eval (jest), wandb (skrypt doinstaluje jeśli brak).
#   - export WANDB_API_KEY=...   (cache ~/.netrc z hosta NIE jest widoczny w kontenerze)
#
# Użycie:
#   export WANDB_API_KEY=<klucz>
#   ./run_pruneonly.sh <checkpoint_dir> <run_name> [gpu_id]
# Przykład:
#   ./run_pruneonly.sh /workspace/nemo_ckpts/qwen3-1.7b-base-depth20 depth20 0
set -euo pipefail

CKPT="${1:?usage: run_pruneonly.sh <checkpoint_dir> <run_name> [gpu_id]}"
NAME="${2:?usage: run_pruneonly.sh <checkpoint_dir> <run_name> [gpu_id]}"
GPU="${3:-0}"

PORT=8080
TASKS="${TASKS:-hellaswag,arc_challenge,winogrande,truthfulqa_mc2}"
WANDB_PROJECT="magisterka-pruning"
RESULTS_DIR="/workspace/results/pruneonly"
NUM_CONCURRENT="${NUM_CONCURRENT:-4}"   # niżej = stabilniej (8 dawało TimeoutError pod obciążeniem)
MAX_INPUT_LEN="${MAX_INPUT_LEN:-8192}"  # 4096 ucinał długie prompty TruthfulQA -> zacięcie serwera
# Osobny cache datasetów dla kontenera — host (inna wersja `datasets`) psuje wspólny cache.
export HF_DATASETS_CACHE="/workspace/hf_datasets_cache_container"

mkdir -p "$RESULTS_DIR"
python -c "import wandb" 2>/dev/null || pip install -q wandb

# --- 0. sprzątnij ewentualne stare serwery (kolizja portów -> 403 triton-access-token) ---
pkill -f "from nemo.collections.llm import deploy" 2>/dev/null || true
pkill -f tritonserver 2>/dev/null || true
sleep 3

# --- 1. deploy w tle (legacy_ckpt=True KLUCZOWE dla sprunowanych) ---
DEPLOY_LOG="$RESULTS_DIR/deploy_${NAME}.log"
echo ">> deploy $NAME (gpu $GPU) -> $DEPLOY_LOG"
CUDA_VISIBLE_DEVICES="$GPU" nohup python -c \
  "from nemo.collections.llm import deploy; deploy(nemo_checkpoint='${CKPT}', num_gpus=1, legacy_ckpt=True, max_input_len=${MAX_INPUT_LEN})" \
  > "$DEPLOY_LOG" 2>&1 &
DEPLOY_PID=$!

cleanup() { kill "$DEPLOY_PID" 2>/dev/null || true; pkill -f tritonserver 2>/dev/null || true; }
trap cleanup EXIT

# --- 2. czekaj na gotowość serwera (max ~4 min) ---
echo ">> czekam na 'Application startup complete'..."
for _ in $(seq 1 120); do
  grep -q "Application startup complete" "$DEPLOY_LOG" 2>/dev/null && { echo ">> serwer gotowy"; break; }
  kill -0 "$DEPLOY_PID" 2>/dev/null || { echo "!! deploy padł, patrz $DEPLOY_LOG"; tail -20 "$DEPLOY_LOG"; exit 1; }
  sleep 2
done
grep -q "Application startup complete" "$DEPLOY_LOG" || { echo "!! timeout na deploy"; exit 1; }

# --- 3. eval przez endpoint (loglikelihood + opcjonalne wandb) ---
# wandb tylko jeśli klucz ustawiony — inaczej eval leci bez logowania (zamiast crashować).
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARGS=(--wandb_args "project=${WANDB_PROJECT},name=${NAME}")
else
  echo "!! WANDB_API_KEY nie ustawiony — eval BEZ logowania do wandb"
fi

echo ">> eval $NAME: $TASKS (concurrent=$NUM_CONCURRENT)"
PYTHONUNBUFFERED=1 lm_eval --model local-completions \
  --model_args "model=triton_model,base_url=http://0.0.0.0:${PORT}/v1/completions/,tokenizer_backend=huggingface,tokenizer=${CKPT}/context/nemo_tokenizer,num_concurrent=${NUM_CONCURRENT},max_retries=5,tokenized_requests=False" \
  --tasks "$TASKS" \
  "${WANDB_ARGS[@]}" \
  --output_path "$RESULTS_DIR/native_suite_${NAME}" \
  2>&1 | tee "$RESULTS_DIR/native_suite_${NAME}.log"

echo ">> gotowe: $NAME (cleanup serwera przez trap EXIT)"
