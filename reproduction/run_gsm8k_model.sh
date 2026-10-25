#!/usr/bin/env bash
set -euo pipefail

model_name=${1:?usage: run_gsm8k_model.sh MODEL_NAME [ENS_BATCH_SIZE]}
ens_batch_size=${2:-1}
repo=/root/autodl-tmp/UQAC
model_path=/root/autodl-tmp/models/${model_name}
base=${repo}/output/gsm8k/${model_name}
sc_base=${repo}/output-sc/gsm8k/${model_name}

cd "${repo}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

common=(
  --model_name_or_path "${model_path}"
  --dataset_dir ./datasets
  --dataset_name gsm8k
  --disable_dataset_cache true
  --dataset_num_mapping_proc 8
  --disable_seq_length_filter false
  --max_seq_length 2048
  --max_new_tokens 1024
  --fp16 false
  --bf16 true
  --use_4bit_quantization false
  --bnb_4bit_compute_dtype bfloat16
  --report_to none
)

if [[ ! -s "${base}/results.json" ]]; then
  python run/main.py "${common[@]}" --output_dir ./output \
    --resp_file results.json --task infer-vllm
fi

if [[ ! -d "${base}/attrs" ]] || [[ $(find "${base}/attrs" -name attrs.h5 2>/dev/null | wc -l) -lt 1200 ]]; then
  python run/main.py "${common[@]}" --output_dir ./output \
    --resp_file results.json --dataset_sorted_by_length true \
    --attrs_save_frequency 500 --attention_reduction max \
    --n_attention_heads 16 --backtracking_buffer_size 3 \
    --backtracking_threshold 0.5 --per_device_eval_batch_size 1 \
    --task infer-uq
fi

python run/main.py "${common[@]}" --output_dir ./output \
  --dataset_sorted_by_length true --per_device_eval_batch_size "${ens_batch_size}" \
  --top_k_similarity 10 --similarity_threshold 0 --task infer-ens

python run/eval/eval.py \
  --model_name_or_path "${model_path}" --output_dir ./output \
  --dataset_dir ./datasets --dataset_name gsm8k \
  --top_k_similarity 10 --similarity_threshold 0 --report_to none

if [[ ! -s "${base}/results_vu.json" ]]; then
  python run/main.py "${common[@]}" --output_dir ./output \
    --resp_file results.json --task infer-vu
fi
python run/eval/eval-vu.py \
  --model_name_or_path "${model_path}" --output_dir ./output \
  --dataset_dir ./datasets --dataset_name gsm8k --report_to none

for seed in 0 1 2 3 4; do
  if [[ ! -s "${sc_base}/results-${seed}.json" ]]; then
    python run/main.py "${common[@]}" --output_dir ./output-sc \
      --resp_file "results-${seed}.json" --inference_temperature 0.5 \
      --seed "${seed}" --logprobs 0 --task infer-vllm
  fi
done
python run/eval/eval-sc.py \
  --model_name_or_path "${model_path}" --output_dir ./output-sc \
  --dataset_dir ./datasets --dataset_name gsm8k --report_to none

echo "COMPLETED ${model_name}"
