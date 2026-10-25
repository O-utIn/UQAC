#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/UQAC
mkdir -p reproduction/state reproduction/logs
export UQAC_INFER_UQ_MAX_GPU_GIB=12

models=(
  Llama-3.2-1B-Instruct
  Llama-3.2-3B-Instruct
  Meta-Llama-3.1-8B-Instruct
  gemma-2-2b-it
  gemma-2-9b-it
  Qwen2.5-1.5B-Instruct
  Qwen2.5-3B-Instruct
  Qwen2.5-7B-Instruct
  DeepSeek-R1-Distill-Llama-8B
)

for dataset in math bbh; do
  for model in "${models[@]}"; do
    if [[ "${model}" == "DeepSeek-R1-Distill-Llama-8B" ]]; then
      export UQAC_INFER_UQ_MAX_GPU_GIB=8
      export UQAC_OFFLOAD_ATTN_OUTPUTS=1
    else
      export UQAC_INFER_UQ_MAX_GPU_GIB=12
      unset UQAC_OFFLOAD_ATTN_OUTPUTS
    fi
    echo "$(date -Is) starting ${dataset} ${model}"
    success=false
    for attempt in 1 2 3; do
      if ./reproduction/run_dataset_model.sh "${dataset}" "${model}" 1; then
        success=true
        break
      fi
      echo "$(date -Is) attempt ${attempt} failed for ${dataset} ${model}" >&2
      sleep 5
    done
    if [[ "${success}" != true ]]; then
      echo "$(date -Is) giving up after 3 attempts: ${dataset} ${model}" >&2
      exit 1
    fi
  done
done

python - <<'PY'
from pathlib import Path

root = Path('/root/autodl-tmp/UQAC')
models = [
    'Llama-3.2-1B-Instruct',
    'Llama-3.2-3B-Instruct',
    'Meta-Llama-3.1-8B-Instruct',
    'gemma-2-2b-it',
    'gemma-2-9b-it',
    'Qwen2.5-1.5B-Instruct',
    'Qwen2.5-3B-Instruct',
    'Qwen2.5-7B-Instruct',
    'DeepSeek-R1-Distill-Llama-8B',
]
missing = []
for dataset in ('gsm8k', 'math', 'bbh'):
    for model in models:
        required = [
            root / 'output' / dataset / model / 'metrics.csv',
            root / 'output' / dataset / model / 'metrics-vu.csv',
            root / 'output-sc' / dataset / model / 'metrics.csv',
        ]
        required += [
            root / 'output-sc' / dataset / model / f'results-{seed}.json'
            for seed in range(5)
        ]
        missing.extend(str(path) for path in required if not path.is_file() or path.stat().st_size == 0)
if missing:
    raise SystemExit('Missing required full-reproduction artifacts:\n' + '\n'.join(missing))
(root / 'reproduction' / 'state' / 'full_reproduction_verified.complete').touch()
PY

echo "$(date -Is) all three datasets and nine models verified complete"
