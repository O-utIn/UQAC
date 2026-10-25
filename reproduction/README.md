# Complete UQAC baseline reproduction on RTX 4090

This directory records the complete reproduction based on public UQAC commit
`610e9a436e043e0643da08f9ff84201d5a68a2e1`.

## Experiment matrix

- Hardware: one NVIDIA RTX 4090 (24 GB)
- Precision: BF16, without quantization
- Datasets: GSM8K, MATH, and BBH
- Models: Llama-3.2-1B/3B-Instruct, Llama-3.1-8B-Instruct,
  Gemma-2-2B/9B-IT, Qwen2.5-1.5B/3B/7B-Instruct, and
  DeepSeek-R1-Distill-Llama-8B
- Metrics: AUROC and 20-bin ECE
- Self-Consistency: temperature `0.5`, seeds 0–4, matching the paper
- Sequence limits: MATH 1,024; BBH 1,536; DeepSeek receives 512 additional
  generation tokens

All 27 dataset/model combinations completed. The repository includes 81 metric
files: `metrics.csv`, `metrics-vu.csv`, and Self-Consistency `metrics.csv` for
each combination. The corresponding 27 paper-format LaTeX tables are in
`tables10-36/`.

## Run

Install the RTX 4090 environment and start or resume the matrix:

```bash
pip install -r requirements-4090.txt
bash reproduction/run_full_reproduction.sh
```

The scripts preserve `/root/autodl-tmp/UQAC` and `/root/autodl-tmp/models` as
the paths used on the experiment server. Replace those hard-coded paths when
running elsewhere. Completed artifacts are detected and skipped, so the
launcher is resumable. `run_full_reproduction.sh` is the final MATH/BBH batch
and all-artifact verifier; the GSM8K stage uses `run_gsm8k_model.sh`.

## RTX 4090 compatibility changes

The public code was adjusted without changing metric definitions:

1. BF16 model loading and CPU offload avoid 24 GB GPU out-of-memory failures.
2. Attention tensors are stacked on CPU and processed in chunks.
3. HDF5 attention-chain output supports safe resume after interruption.
4. `textspan.py` provides Python 3.12 compatibility where `pytextspan` has no
   wheel.
5. Self-Consistency scripts use the actual `--resp_file` argument and
   `run/eval/eval-sc.py` entry point.
6. The dependency set in `requirements-4090.txt` resolves the incompatible
   NumPy/Matplotlib/scikit-learn pins in the upstream environment.

## Results layout

```text
results/
  output/<dataset>/<model>/metrics.csv
  output/<dataset>/<model>/metrics-vu.csv
  output-sc/<dataset>/<model>/metrics.csv
reproduction/tables10-36/*.tex
```

`baseline_qwen1.5_gsm8k.csv` and `.tex` are retained as an early single-model
checkpoint; the authoritative complete results are under `results/`.
