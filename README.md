# TheGeometryOfFormalLogic
The Geometry of Formal Logic: Linear Probing of Learn 4 Tactic States in Mathematical Language Models

## Mathematical Concepts
### Lean Type Hierarchy
#### Core `Sort u` Ladder
Everything in Lean has a type, and those types also have types. This creates an infinite ladder called **Universes**. The fundamental building block is `Sort u`, where `u` is the universe level.
|Term|Its Type|Short Name|Universe Level|
|----|--------|----------|--------------|
|`True`,`2+2=4`|`Prop`|`Sort 0`|`0`|
|`Nat`,`String`,`List`,`Int`|`Type`|`Sort 1`|`1`|
|`Type`|`Type 1`|`Sort 2`|`2`|
|`Type 1`|`Type 2`|`Sort 3`|`3`|

The most critical boundary comes from the difference between `Prop` and `Type`:
`Prop` is of `Sort 0`, and is the world of logical statements. Objects in Prop follow Proof Irrelevance. This means if you have two different proofs of $2+2=4$, Lean’s kernel treats them as identical. Geometrically, you might expect the LLM to collapse all proof-steps of the same proposition into a single point.

`Type` is of `Sort 1+`, and is the world of data structures. Here objects are relevant, since `5` is distinctly different than `6`.
#### The Curry-Howard Correspondence
This maps logic onto computation. When probing a Large Language Model, we are checking for this mapping
|Logic Side|Computation Side|Lean Example|
|----------|---------------|-------------|
|**Proposition** ($P$)|**Type** ($T$)|`n>5`|
|**Proof** ($p$)|**Term/Program** ($t$)|`h : n>5`|
|**Implication** ($P\rightarrow Q$)|**Function** ($A\rightarrow B$)|`fun h => ...`|
|**Conjunction** ($P $land Q$)|**Product Type** ($A\times B$)|`And.intro h1 h2`|
|**Disjunction** ($P \lor Q$)|**Sum Type** ($A\oplus B$)|`Or.inl h1`|

If the model understands math, its embeddings for a "Proof" should geometrically resemble its embeddings for a "Function."

#### Universe Polymorphism
Mathematic often requires theorems across levels. Lean uses universe variables to deal with this. For example, a `List` can contain integers of `Type 0` or it can contain other types `Type 1`. Instead of defining `List` 100 times, Lean instead defines it once as `List.{u} (alpha : Type u) : Type u`

In the type hierarchy of the models we are testing, we are looking for the `Prop` Hyperplane, Definitional Equality, and Impredicativity.
- The `Prop` Hyperplane lets a linear probe distinguish between a term that is a Data Type and one that is a Proposition.
- Definitional Equality in lean means that `3` and `(fun x => x + 1) 2` are definitionally equal. Does the LLMs embedding put these two close together?
- Impredicativity means a proposition can talk to all other propositions but a `Type u` cannot contain all `Type u`'s.

Our **smoking gun** would be if we find that the probe can predict the universe level of a term with high accuracy in the middle layers. It would be a strong sign that the model has internalized the hierarchical nature of formal logic. 
### Tactic Theory
In Lean, there are two ways to write a proof: tactic mode and tem mode. In Term mode, you write a full mathematical expression, like typing a line of code, e.g. `hp : p`, `hq: q`. In Tatcic mode, one uses the `by` keyword to enter an interactive environment. 

#### The Tactic State (The Goal)
When one is in tactic mode, Lean keeps track of the `Tactic State`. This is the data to be probed in the research. A state consists of **Local Context**, or the hypothesis we currently have, and **The Target**, or the thing we are trying to prove (denoted `$\vdash$).
#### How Tactics Manipulate States
Tactics are the backwards-reasoning tools used to transform the current goal into zero or more subgoals.
|Tactic|Programming Analogy|Effect on Tactic State|
|------|-------------------|----------------------|
|`intro h`|Naming a function parameter|Moves a premise from the goal $\vdash$ `P` $\rightarrow$ `Q` into the context as `h : P`.|
|`apply f`|Calling a function|If `f : P -> Q` and the goal is `Q`, it changes the goal to `P` (To prove Q, I now only need to prove P)|
|`rw [h]`|Find and replace|Uses an equality `h : a=b` to swap `a`'s for `b`'s.|
|`exact h`|`return` statement|Closes the goal because `h` was exactly the proof we needed.|
|`simp`|Auto-refactoring|Runs a suite of lemmas to simplify the expression into normal form.|

#### The Hidden Proof Term
Tactics do not exist in the final proof. When one finishes a tactic proof, Lean's **Elaborator** takes the commands and compiles them into a single Proof Term, which is a lambda expression. So, if the tactic script was `by intro h; exact h`, then the resulting term will be `fun h => h`.

The tactic language has combiners:
- `<;>` (Chain): Applies the second tactic to all subgoals created by the first
- `try`/`repeat`: Allows for automation and searching
- `all_goals`: Forces a tactic to run on every open problem simultaneously.
### Vector Space Basis & Orthogonality
### Cosine Similarity & L2 Norm
## Machine Learning
### Linear Regression
### Logistic Regression
### Train/Test splitting for proofs
### PCA & t-SNE & UMAP
## Transformer Architecture
### The Residual Stream
### Layer-wise Evolution
### Byte-Pair Encoding
## Mechanistic Interpretability
### Linear Representation Hypothesis
### Logit Lens & Tuned Lens
### Activation Steering & Saliency
### Superposition
## Tooling & Data Alignment
### LeanDojo
### TransformerLens
### JSON
### State $S_i$ (Lean Goal) $\rightleftarrow$ Activation $A_i$ (Hidden State)
