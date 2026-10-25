# Reproduction metrics

This directory contains the compact metrics from the complete RTX 4090
reproduction. Raw generations, model weights, HDF5 attention tensors, and
datasets are intentionally excluded because they are large and reproducible
from the included scripts.

For every dataset/model pair:

- `output/.../metrics.csv`: token-probability and UQAC metrics
- `output/.../metrics-vu.csv`: verbalized-uncertainty metrics
- `output-sc/.../metrics.csv`: Self-Consistency metrics at temperature 0.5

There are 27 dataset/model pairs and 81 CSV files in total. See
[`../reproduction/README.md`](../reproduction/README.md) for the exact protocol
and [`table1-paper-vs-reproduction.md`](table1-paper-vs-reproduction.md) for a
compact comparison with the main paper table.
