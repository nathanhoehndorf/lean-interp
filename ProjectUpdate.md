# The Geometry of Formal Logic: Probing the Linear Representation Hypothesis in Lean 4 LLMs
- **Author**: Nathan Hoehndorf
- **MATH498-C**

## Abstract
Large Language Models (LLMs) have demonstrated emergent capabilities in formal mathematics, yet the internal geometry of these representations remains under-explored. This project investigates whether LLMs trained on or exposed to Lean 4 represent underlying mathematical concepts linearly in their activation space, or if they rely on alternative structures such as hyperbolic representations or simple memorization. Preliminary experiments utilizing a linear probe on the residual stream (Layer 18) of Qwen2.5-1.5B yielded a Mean Squared Error (MSE) of 116.9 when predicting proof state variables, indicating a lack of linear encoding in the tested model. Future work will expand probing to out-of-distribution domains, execute full layer sweeps, and utilize compute credits to test models explicitly fine-tuned on Lean 4.

## Introduction
The Linear Representation Hypothesis posits that neural networks encode high-level semantic concepts as approximately linear directions within their activation space, enabling downstream recovery via simple linear probes (Alain & Bengio, 2016; Elhage et al., 2021; Nanda et al. 2023). However, probing methods are known to have linear classifiers recover a signal even from representations that don't explicitly encode the feature, necessitating control tasks and careful evaluation design (Hewitt & Liang, 2019).

The primary question this research seeks to answer is: Do Large Language Models trained on Lean 4 represent the mathematical ideas encoded into the language of Lean 4 linearly in their activation space? Procuring evidence for this hypothesis would suggest that LLMs are building abstractions of formal logic rather than relying on stochastic parrot-like mimicry. Conversely, a lack of linear representation opens the door to alternative hypotheses, namely that mathematical relationships in LLMs may instead be embedded in non-Euclidean geometries such as hyperbolic space, which has been shown to efficiently represent hierarchical and tree-like structures (Nickel & Kiela, 2017; Sala et al., 2018).

Prior work has demonstrated that language models can internalize aspects of formal reasoning and theorem proving, though the structure of these internal representations remains poorly understood (Polu & Sutskever, 2020; Lewkowycz et al., 2022).

## Codebase Map
So far, the entire project is just at the root level because I have just barely gotten enough stuff down to need it. So, here is the map of the architecture:
```bash
.
├── README.md
├── ProjectUpdate.md
├── main.py
├── extract_activations.py
├── extract_data.py
├── probe.py
├── model.py
├── loader.py   
└── gold_dataset.json
```

## Methodology
### Experimental Setup
The current pipeline utilizes the `HookedTransformer` library to extract activations from intermediate layers. Because hardware constraints limit continuous inference, the architecture is designed to cache activations to disk, allowing for rapid, iterative training of the linear probe. In the future, this may change if I need to use compute credits.

### Feature Selection for Probing
To determine the extent of linear representation, the probe must test various dimensions of a proof state. The current pipeline evaluates Variable Count (the number of active variables in the context). Future experiments will test the following 7 mathematical features:

1. **Goal Depth/Complexity**: The number of nested implications or operators in the target `⊢` goal.
2. **Quantifier Presence**: Binary classification, whether or not proof state relies on universal (`$\forall$`) or existential (`$\exists$`) quantifiers.
3. **Context Size**: The raw number of hypotheses currently loaded into the local context.
4. **Type Complexity**: Classification of the highest-order type present.
5. **Inductive vs. Direct**: Binary classification of whether the current state is inside an inductive step.
6. **Equality vs Inequality**: Whether the primary goal is proving an equivalence or a bound.
7. **Unbound Variables**: The count of variables in the goal that have not yet been instantiated.

### Experimental Controls and Baselines
To validate that the linear probe is identifying mathematical abstractions rather than syntatic noise or statistical artifacts, the following controls are implemented:

### Shuffled Label Control (Random Baseline)

We establish a lower-bound performance metric by training a linear probe on a version of the dataset where the mathematical features (labels) have been randomly shuffled.

The purpose of this is to test our Null Hypothesis. If the probe trained on real labels does not significantly outperform the shuffled-label probe, we conclude that the model's residual stream contains no recoverable linear signal for that feature.

### Syntactic Control Features (Token Length Correlation)

LLMs often correlate complexity with sequence length. To ensure the probe is not simply regressing on the number of tokens in a proof state, we introduce a **Control Feature**: a meaningless metric that is highly correlated with input size, like total character count or number of whitespace characters.

By comparing the probe's performance on Variable Count vs Character Count, we can determine if the model is truly abstracting variables as a distinct logical category or merely reflecting the physical length of the prompt.

### Representation Controls (Layer & Embedding Baselines)

To ensure the mathematical signal is a product of deeper model computation, we apply the probe to two non-logical representations:
1. **Early Layer Probing**: We probe the initial embeddings (Layer 0) before any transformer blocks have processed the logic. Accuracy here would suggest the feature is trivial and purely lexical.
2. **Unrelated Model Weights**: We probe the same features using the weights of a model of similar size that was never exposed to Lean 4 or mathematical data (e.g., a base language model). This identifies whether the "geometry" is a universal property of language models or a specific result of formal logic training.

### Invariance Testing (Cross-Domain Validation)

We evaluate the probe's gap in generalization by training on one logical domain and testing on another. A truly mathematical representation of a variable should be invariant to the domain. If a probe trained on `Nat` variables fails to identify `List` variables, it suggest the model has learned domain-specific syntax rather than a generalized mathematical concept.

## Preliminary Results
Initial probing was conducted on Layer 18 of the Qwen2.5-1.5B model, attempting to linearly regress the number of variables in 500 Lean 4 proof states.

The probe converged at a Mean Squared Error (MSE) of 116.9. In the context of variable counts, an error of this magnitude is mathematically indistinguishable from random chance.

These results prompted a re-evaluation of the current model scale. It is highly likely that a generalized 1.5B model, despite extensive pre-training, lacks the requisite depth to form linear abstractions of Lean 4. This heavily implies that either the representation is non-linear (e.g., hyperbolic), or that specific Lean-4 fine-tuned models and larger parameter counts are strictly necessary to observe the geometry of formal logic. We move forward by experimenting with the latter.

## Remaining Experiments
To conclusively answer the overarching research question, the following experiments remain, at minimum:
- **Full Layer Sweeps**: Probing across all layers to locate where abstractions might emerge before being projected back into vocabulary space.
- **Out-of-Distribution Generalization**: Training the probe on logic proofs and evaluating its accuracy on data structure proofs to test if the representation of "variable" is universal or domain-dependent, for example.
- **Multi-Feature Probing**: Implementing the expanded feature list detailed in Section 3.2.
- **Model Scaling**: Transitioning to models explicitly trained on Lean 4, utilizing compute credits to overcome current hardware bottlenecks if necessary.

## Roadblocks

The primary roadblock to date has been hardware constraints. Extracting activations from large, specialized models requires significant VRAM, forcing the initial pilot tests onto smaller models running on CPU. This drastically slows down dataset processing and risks generating false negatives regarding the linear representation hypothesis simply due to inadequate model capacity.

## Questions for Dr. Ivanitskiy

1. **Avoiding Statistical Bias**: As I transition to testing more complex features across different models, how can I best ensure I am choosing appropriate training data and evaluation metrics? I want to avoid being a "lazy statistician" who unintentionally p-hacks or structures the linear probes in a way that just confirms the hypothesis I want to see.

2. **Model Selection:** Given the preliminary failure on a generalized 1.5B model, do you have recommendations for specific open-weight models that have a known, robust density of formal math/Lean 4 training data that I should prioritize when I deploy my compute credits?

3. **Hyperbolic vs. Linear**: If the representation of proof states is indeed hyperbolic (due to the tree-like nature of logical deduction), are there computationally feasible ways to probe for hyperbolic geometry within this project's timeframe, or should I strictly limit my scope to proving/disproving the linear hypothesis?

## Informal Bibliography
- Alain, G., & Bengio, Y. (2016). Understanding intermediate layers using linear classifier probes.
- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. Anthropic.
- Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks.
- Lewkowycz, A., et al. (2022). Solving Quantitative Reasoning Problems with Language Models.
- Nickel, M., & Kiela, D. (2017). Poincaré Embeddings for Learning Hierarchical Representations.
- Nanda, N., et al. (2023). Progress Measures for Grokking via Mechanistic Interpretability.
- Polu, S., & Sutskever, I. (2020). Generative Language Modeling for Automated Theorem Proving.
- Sala, F., et al. (2018). Representation Tradeoffs for Hyperbolic Embeddings.