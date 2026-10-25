# Paper Table 1 vs. RTX 4090 reproduction

The full per-dataset/per-model values are preserved in the 81 CSV files and 27
paper-format LaTeX tables. This index records the comparison scope and avoids
silently mixing the initial temperature-0.3 checkpoint with the final run.

| Item | Paper protocol | This reproduction |
|---|---|---|
| Datasets | GSM8K, MATH, BBH | GSM8K, MATH, BBH |
| Models | 9 models | Same 9 models |
| Dataset/model runs | 27 | 27 complete |
| Self-Consistency temperature | 0.5 | 0.5 |
| Self-Consistency samples | 5 | 5 (seeds 0–4) |
| Metrics | AUROC / ECE | AUROC / ECE |
| Precision | BF16 | BF16 |
| Hardware | Not constrained to one 24 GB GPU | One RTX 4090, 24 GB |

For numeric, paper-formatted comparisons, open the matching file in
[`../reproduction/tables10-36/`](../reproduction/tables10-36/). Each table is
generated directly from the corresponding CSV artifacts in this repository.
