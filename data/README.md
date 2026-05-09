---
dataset_info:
  features:
  - name: url
    dtype: string
  - name: commit
    dtype: string
  - name: file_path
    dtype: string
  - name: full_name
    dtype: string
  - name: start
    sequence: int64
  - name: end
    sequence: int64
  - name: traced_tactics
    dtype: string
  splits:
  - name: train
    num_bytes: 320023872
    num_examples: 98514
  - name: test
    num_bytes: 6116916
    num_examples: 2000
  - name: validation
    num_bytes: 7228697
    num_examples: 2000
  download_size: 54194769
  dataset_size: 333369485
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: validation
    path: data/validation-*
---
# Dataset Card for "lean-dojo-mathlib4"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)