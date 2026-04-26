# The Geometry of Formal Logic
**Nathan Hoehndorf | Colorado School of Mines**

This repository contains the infrastructure for a Mechanistic Interpretability study probing how LLMs represent Lean 4 formal logic states.

The most up-to-date progress report is the file `ProjectUpdate.md`.

## Project Structure
- **`extract_data.py`**: The "Data Harvester." Uses LeanDojo to extract tactic states and generates labels (Variable Count).
- **`extract_activations.py`**: Contains the logic for capturing the residual stream from specific transformer layers using `transformer_lens`.
- **`probe.py`**: Implements the linear probing architecture and MSE loss training loop.
- **`model.py` / `loader.py`**: Utilities for loading both general (Pythia, Llama) and Lean-specialized (ReProver) models.
- **`main.py`**: The central execution script to run the full pipeline from data loading to probe evaluation.
- **`gold_dataset.json`**: The harvested dataset containing tactic states, their variable count labels, and their domain (e.g., Logic, Time).

## Getting Started
1. **Setup**: This project uses `uv` for Python 3.12 dependency management.
   ```bash
   uv sync
   uv run extract_data.py
   uv run main.py

## Experimental Goals
We're testing the Linear Representation Hypothesis for logical features, comparing trained weights against a randomly initialized baseline, and we're analyzing weight sparsity to determine entanglement of logical concepts.