# The Geometry of Formal Logic
**Nathan Hoehndorf | Colorado School of Mines**

This repository contains the infrastructure for a Mechanistic Interpretability study probing how LLMs represent Lean 4 formal logic states.

## Repository Structure

- `TheGeometryOfFormalLogic/`: Main source code for the project.
  - `main.py`: End-to-end probing workflow.
  - `config.py`: Configuration and CLI argument handling.
  - `extract_activations.py`: Tools for extracting model activations.
  - `loader.py`: Dataset loading utilities.
  - `model.py`: Model loading and initialization.
- `data/`: Experimental data, logs, and raw results.
  - `results/`: JSON result files from probing runs.
  - `data/`: Parquet files used for training/validation.
  - `gold_dataset.json`: The processed dataset used for probing (53MB).
  - *Large datasets (>100MB) such as `states.jsonl` and `activations_cache_*.pt` are excluded from this repository due to size constraints.*
- `docs/`: Project documentation and writeups.
  - `ProjectUpdate.md`: Latest progress report and methodology.
  - `figures/`: Figures and plots generated from experiments.
- `notebooks/`: Jupyter notebooks for exploratory analysis (currently empty).

## Usage

The project uses `uv` for dependency management. To run the probing workflow:

```bash
uv run TheGeometryOfFormalLogic/main.py --mode control --sweep-layers
```

By default, the results and figures will be saved in `data/results/` and `docs/figures/` (configured via `config.py`).

## Datasets

The primary dataset used is `gold_dataset.json`. Larger raw files and activation caches are managed locally and ignored by git.

## Writeup

The project writeup and results can be found in `docs/TheGeometryofProof.pdf`.
