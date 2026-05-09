# The Geometry of Formal Logic
**Nathan Hoehndorf | Colorado School of Mines**

This repository contains the infrastructure for a Mechanistic Interpretability study probing how LLMs represent Lean 4 formal logic states.

The most up-to-date progress report is the file `ProjectUpdate.md`.

## MVP Usage

The repository now includes an end-to-end probing workflow in `main.py` with layer-wise figures and binary feature metrics.

Example control run using a text-trained GPT-2 model:

```bash
python main.py --mode control --sweep-layers
```

Example tester run using a Lean-trained model:

```bash
python main.py --mode tester --sweep-layers
```

By default, control mode uses the local `./model_weights/gpt2` checkpoint and tester mode uses `RickyDeSkywalker/TheoremLlama`.

Next Steps:
1. Binary Classifier, Logistic Regression, Linear Support Vector Machine (separates activations based on a binary property)
2. Mass-Mean, Mean-Difference ()
