#!/usr/bin/env bash
set -u

repo=/root/autodl-tmp/UQAC
models=/root/autodl-tmp/models
state=${repo}/reproduction/state
mkdir -p "${state}" "${repo}/reproduction/logs"
cd "${repo}"

log() { printf '%s %s\n' "$(date -Is)" "$*"; }

run_retry() {
  local label=$1
  shift
  local attempt
  for attempt in 1 2 3; do
    log "${label}: attempt ${attempt}/3"
    if "$@"; then
      log "${label}: complete"
      return 0
    fi
    log "${label}: failed; retrying in 60 seconds"
    sleep 60
  done
  log "${label}: FAILED after 3 attempts"
  return 1
}

wait_for_model() {
  local model=$1
  local marker=${2:-}
  [[ "${marker}" == "-" ]] && marker=""
  while [[ ! -s "${models}/${model}/config.json" ]] ||
        { [[ -n "${marker}" ]] && [[ ! -f "${state}/${marker}" ]]; }; do
    log "waiting for model ${model}"
    sleep 60
  done
}

chmod +x reproduction/run_sc_gsm8k.sh reproduction/run_gsm8k_model.sh

run_retry "SC-0.5 Llama-3.2-3B-Instruct" \
  ./reproduction/run_sc_gsm8k.sh Llama-3.2-3B-Instruct 0.5 || true

while read -r model batch marker; do
  wait_for_model "${model}" "${marker}"
  run_retry "full GSM8K ${model}" \
    ./reproduction/run_gsm8k_model.sh "${model}" "${batch}" || true
done <<'MODELS'
Meta-Llama-3.1-8B-Instruct 1 -
gemma-2-2b-it 4 -
gemma-2-9b-it 1 -
Qwen2.5-7B-Instruct 1 Qwen2.5-7B-Instruct.downloaded
DeepSeek-R1-Distill-Llama-8B 1 DeepSeek-R1-Distill-Llama-8B.downloaded
MODELS

touch "${state}/gsm8k_all_models.complete"
log "all GSM8K model runs complete"
