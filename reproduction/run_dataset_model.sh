#!/usr/bin/env bash
set -euo pipefail

dataset=${1:?usage: run_dataset_model.sh DATASET MODEL_NAME [ENS_BATCH_SIZE]}
model_name=${2:?usage: run_dataset_model.sh DATASET MODEL_NAME [ENS_BATCH_SIZE]}
ens_batch_size=${3:-1}

case "${dataset}" in
  math)
    # The paper's 1,024-token limit is the generated response limit.  The
    # repository uses a 2,048-token model context so VU can append its prompt
    # to that response without crossing vLLM's hard context boundary.
    max_seq_length=2048
    max_new_tokens=1024
    uq_seq_length=1024
    ;;
  bbh)
    max_seq_length=2048
    max_new_tokens=1536
    uq_seq_length=1536
    ;;
  *)
    echo "unsupported dataset: ${dataset}" >&2
    exit 2
    ;;
esac

# The paper adds 512 tokens for DeepSeek-R1's thinking tokens.
if [[ "${model_name}" == "DeepSeek-R1-Distill-Llama-8B" ]]; then
  max_seq_length=$((max_seq_length + 512))
  max_new_tokens=$((max_new_tokens + 512))
  uq_seq_length=$((uq_seq_length + 512))
fi

repo=/root/autodl-tmp/UQAC
model_path=/root/autodl-tmp/models/${model_name}
base=${repo}/output/${dataset}/${model_name}
sc_base=${repo}/output-sc/${dataset}/${model_name}
uq_marker=${repo}/reproduction/state/${dataset}.${model_name}.uq.complete

cd "${repo}"
mkdir -p reproduction/state
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

common=(
  --model_name_or_path "${model_path}"
  --dataset_dir ./datasets
  --dataset_name "${dataset}"
  --disable_dataset_cache true
  --dataset_num_mapping_proc 8
  --disable_seq_length_filter false
  --max_seq_length "${max_seq_length}"
  --max_new_tokens "${max_new_tokens}"
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

if [[ ! -f "${uq_marker}" ]]; then
  python run/main.py "${common[@]}" --output_dir ./output \
    --resp_file results.json --dataset_sorted_by_length true \
    --max_seq_length "${uq_seq_length}" \
    --attrs_save_frequency 500 --attention_reduction max \
    --n_attention_heads 16 --backtracking_buffer_size 3 \
    --backtracking_threshold 0.5 --per_device_eval_batch_size 1 \
    --task infer-uq
  touch "${uq_marker}"
fi

if [[ ! -s "${base}/metrics.csv" ]]; then
  python run/main.py "${common[@]}" --output_dir ./output \
    --dataset_sorted_by_length true --per_device_eval_batch_size "${ens_batch_size}" \
    --max_seq_length "${uq_seq_length}" \
    --top_k_similarity 10 --similarity_threshold 0 --task infer-ens

  python run/eval/eval.py \
    --model_name_or_path "${model_path}" --output_dir ./output \
    --dataset_dir ./datasets --dataset_name "${dataset}" \
    --top_k_similarity 10 --similarity_threshold 0 --report_to none
fi

if [[ ! -s "${base}/results_vu.json" ]]; then
  python run/main.py "${common[@]}" --output_dir ./output \
    --resp_file results.json --task infer-vu
fi
if [[ ! -s "${base}/metrics-vu.csv" ]]; then
  python run/eval/eval-vu.py \
    --model_name_or_path "${model_path}" --output_dir ./output \
    --dataset_dir ./datasets --dataset_name "${dataset}" --report_to none
fi

for seed in 0 1 2 3 4; do
  if [[ ! -s "${sc_base}/results-${seed}.json" ]] || \
     [[ ! -s "${sc_base}/results-${seed}_logprobs.pt" ]]; then
    python run/main.py "${common[@]}" --output_dir ./output-sc \
      --resp_file "results-${seed}.json" --inference_temperature 0.5 \
      --seed "${seed}" --logprobs 0 --task infer-vllm
  fi
done
if [[ ! -s "${sc_base}/metrics.csv" ]]; then
  python run/eval/eval-sc.py \
    --model_name_or_path "${model_path}" --output_dir ./output-sc \
    --dataset_dir ./datasets --dataset_name "${dataset}" --report_to none
fi

echo "COMPLETED ${dataset} ${model_name}"
