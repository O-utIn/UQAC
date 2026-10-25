#!/usr/bin/env bash
set -u

repo=/root/autodl-tmp/UQAC
models=/root/autodl-tmp/models
mkdir -p "${repo}/reproduction/state" "${repo}/reproduction/logs"

log() { printf '%s %s\n' "$(date -Is)" "$*"; }

log "waiting for the active Qwen2.5-7B-Instruct download"
while pgrep -f '[m]odelscope download --model Qwen/Qwen2.5-7B-Instruct' >/dev/null; do
  sleep 30
done

if [[ -s "${models}/Qwen2.5-7B-Instruct/config.json" ]] &&
   find "${models}/Qwen2.5-7B-Instruct" -name '*.safetensors' -size +100M | grep -q .; then
  touch "${repo}/reproduction/state/Qwen2.5-7B-Instruct.downloaded"
  log "Qwen2.5-7B-Instruct download complete"
else
  log "Qwen2.5-7B-Instruct download incomplete; resuming"
  modelscope download --model Qwen/Qwen2.5-7B-Instruct \
    --local_dir "${models}/Qwen2.5-7B-Instruct"
  touch "${repo}/reproduction/state/Qwen2.5-7B-Instruct.downloaded"
fi

log "downloading DeepSeek-R1-Distill-Llama-8B"
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --local_dir "${models}/DeepSeek-R1-Distill-Llama-8B"
touch "${repo}/reproduction/state/DeepSeek-R1-Distill-Llama-8B.downloaded"
log "all remaining downloads complete"
