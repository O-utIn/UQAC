#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/UQAC
mkdir -p reproduction/state reproduction/logs
# Keep enough GPU headroom for all retained attention matrices.  Eight-billion
# parameter weights fit in 24 GiB, so a 20 GiB ceiling does not trigger layer
# offload and the forward pass still exhausts memory on long GSM8K answers.
export UQAC_INFER_UQ_MAX_GPU_GIB=12

for spec in \
  'Meta-Llama-3.1-8B-Instruct 1' \
  'Qwen2.5-7B-Instruct 1' \
  'DeepSeek-R1-Distill-Llama-8B 1'; do
  read -r model batch <<<"${spec}"
  printf '%s recovering %s\n' "$(date -Is)" "${model}"
  ./reproduction/run_gsm8k_model.sh "${model}" "${batch}"
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
for model in models:
    required = [
        root / 'output' / 'gsm8k' / model / 'metrics.csv',
        root / 'output' / 'gsm8k' / model / 'metrics-vu.csv',
        root / 'output-sc' / 'gsm8k' / model / 'metrics.csv',
    ]
    missing.extend(str(p) for p in required if not p.is_file() or p.stat().st_size == 0)
if missing:
    raise SystemExit('Missing required metrics:\n' + '\n'.join(missing))
(root / 'reproduction' / 'state' / 'gsm8k_verified.complete').touch()
PY

printf '%s all nine models verified complete\n' "$(date -Is)"
