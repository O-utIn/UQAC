#!/usr/bin/env bash
set -euo pipefail

model_name=${1:?usage: run_sc_gsm8k.sh MODEL_NAME [TEMPERATURE]}
temperature=${2:-0.5}
repo=/root/autodl-tmp/UQAC
model_path=/root/autodl-tmp/models/${model_name}
sc_base=${repo}/output-sc/gsm8k/${model_name}

cd "${repo}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

for seed in 0 1 2 3 4; do
  if [[ ! -s "${sc_base}/results-${seed}.json" ]]; then
    python run/main.py \
      --model_name_or_path "${model_path}" \
      --output_dir ./output-sc --resp_file "results-${seed}.json" \
      --dataset_dir ./datasets --dataset_name gsm8k \
      --disable_dataset_cache true --dataset_num_mapping_proc 8 \
      --disable_seq_length_filter false --max_seq_length 2048 \
      --max_new_tokens 1024 --inference_temperature "${temperature}" \
      --fp16 false --bf16 true --use_4bit_quantization false \
      --bnb_4bit_compute_dtype bfloat16 --report_to none \
      --seed "${seed}" --logprobs 0 --task infer-vllm
  fi
done

python run/eval/eval-sc.py \
  --model_name_or_path "${model_path}" --output_dir ./output-sc \
  --dataset_dir ./datasets --dataset_name gsm8k --report_to none

echo "COMPLETED SC temperature=${temperature} ${model_name}"
