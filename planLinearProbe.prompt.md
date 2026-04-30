## Plan: Align Probe Design with Label Granularity

TL;DR - The current code trains a global regression probe on the last-token activation to predict a sequence-level variable count. The proposed issue is valid for token-level supervision, but the current pipeline is not actually a per-position probe. The safe fix is either: keep the global count regression baseline and add a token-level binary classification probe, or refactor the probe entirely to use per-token labels if the goal is variable/keyword recognition.

**Steps**
1. Confirm the exact current pipeline.
   - `main.py` reads `gold_dataset.json` and uses `item['label']` as a scalar count.
   - `extract_activations.py` extracts only `last_token_act` from `blocks.{layer}.hook_resid_post`.
   - `train_probe()` in `main.py` and `probe.py` trains a linear layer with MSE on this scalar.
2. Decide target task and label type.
   - Option A: keep the global task as "variable count" and treat it as a summary regression probe.
   - Option B: switch/add a token-level probe with binary labels such as `is_var`/`is_kwd`.
   - Recommend implementing Option B as a new probe path while keeping the existing baseline.
3. Extend dataset generation.
   - Add token-level annotation code in `extract_data.py` or a new helper file.
   - Tokenize each `state` and annotate each token as variable, keyword, or neither.
   - Store per-token labels in the dataset format (e.g. `tokens`, `is_var`, `is_kwd`).
4. Update activation extraction.
   - Modify `extract_activations.get_activations()` to return activations for all token positions, not just the last token.
   - Add a new dataset preparation path in `prepare_dataset()` that flattens token activations and corresponding token labels.
5. Add a token-level probe trainer.
   - Add a binary classifier in `probe.py` or `main.py` using BCEWithLogitsLoss for `is_var` / `is_kwd`.
   - Optionally add a multi-task probe if both labels are desired simultaneously.
6. Preserve the existing regression pipeline as baseline.
   - Keep `train_probe()` and existing `main.py` path for the global count regression.
   - Add a separate entrypoint or flag for token-level probe experiments.
7. Add verification and evaluation.
   - Check data shape and label alignment for a small sample.
   - Train token classifier and inspect training loss and accuracy/F1.
   - Compare with the baseline global regression MSE.

**Relevant files**
- `/home/nathan/TheGeometryOfFormalLogic/main.py` — currently orchestrates data loading, activation extraction, and regression probe training.
- `/home/nathan/TheGeometryOfFormalLogic/extract_activations.py` — currently extracts only the last token activation; needs per-position support.
- `/home/nathan/TheGeometryOfFormalLogic/extract_data.py` — currently generates scalar variable-count labels; needs token-level annotation.
- `/home/nathan/TheGeometryOfFormalLogic/probe.py` — contains the toy probe training code and can be extended to host token-level probe utilities.

**Verification**
1. Confirm `gold_dataset.json` items contain `state` and scalar `label`, as expected.
2. Generate a small token-level sample and inspect label arrays for correctness.
3. Train the new token-level classifier on the sample and verify loss decreases and labels are aligned.
4. Run existing global regression baseline to ensure no regression breakage.

**Decisions**
- Use global regression only as a baseline; token-level classification is more appropriate for `is_var` / `is_kwd` semantics.
- If keeping count prediction, consider simpler classification buckets instead of raw regression to improve stability.
- Keep the final token activation design for the global probe, since it already summarizes all prior context.

**Further Considerations**
1. Need clarification whether the main research target is "variable count" or "token-level variable identity".
2. If token-level labels are desired, we should decide between raw token text parsing and tokenizer-aligned labels.
3. If the model is autoregressive, the last-token residual is acceptable for global summaries, but per-token tasks should extract every position.