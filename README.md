# TheGeometryOfFormalLogic
The Geometry of Formal Logic: Linear Probing of Learn 4 Tactic States in Mathematical Language Models

* Try circuit analysis using SAE circuits

## To-Do
1. The Data Harvester. * Modify your extract_data.py script to run across a larger chunk of your example repository or Mathlib.Write a simple parsing function to automatically generate your labels ($y$) from the state_before strings.
2. Task 3: Build the Gold Dataset. * Extract and save 1,000 to 5,000 (State, Label) pairs. Split this cleanly into Training (70%), Validation (15%), and Test (15%) sets.
3. Task 4: The Layer Sweep. * Train your linear probe on the training set, but do it for every single layer of Pythia-160m (Layers 0 through 11). Hypothesis: Early layers act like a dictionary and won't know the mathematical properties. Middle/late layers will show a sudden drop in generalization loss as the representation crystallizes.
4. Task 5: The Control Experiment. * Shuffle your $y$ labels randomly and retrain. If the loss still drops, your probe is too large or your dataset is too small. This is crucial for proving you found a real geometric feature, not statistical noise.
5. Task 6: The Transfer Test. * Train a probe to find the "Variable Count" vector on addition problems, then test it on multiplication problems. Does the linear direction generalize across different mathematical operators?
6. Task 7: Dimensionality Reduction. * Extract the activations for 100 different Lean states. Run Principal Component Analysis (PCA) on these 768-dimensional vectors. Do states with the same properties naturally cluster together in the top 2 or 3 principal components?
7. Task 8: Activation Steering (Optional but High-Impact). * Take the weights of your successful linear probe. This is your "Concept Vector." Pass a new Lean state into the model, but artificially add or subtract this vector from the residual stream during the forward pass. Does this causally change the model's next token prediction?
8. Task 9: Data Visualization. * Generate the core plots: Layer-wise accuracy curves, PCA scatter plots of the activation space, and a table comparing your real probe's accuracy versus the randomized control.
9. Drafting the Paper.

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
### Cosine Similarity & L2 Norm
#### L2 (Euclidean) Norm
The L2 norm of a vector $v=[x_1,x_2,\dots,x_n] is its standard length from the origin: \|\|$v_2$\|\|=$\sqrt{\sum_{i=1}^{n}x_i^2}$. In Transformers, the L2 Norm of the residual stream typically grows as one goes deeper into the model. If you are probing a Lean proof, a small L2 norm in the induction direction suggests that the model is unsure or hasn't processes that logic yet. A large L2 norm suggest that a model is committed to a mathematical path.
#### Cosine Similarity
Asks how much two vectors points in the same direction. It measure the cosine of the angle between the two vectors: $\cos(\theta)=\frac{A\cdot B}{\lVert A\rVert_2\lVert B\rVert_2}$. If we have a vector for the concept "Commutative Property" and a vector for the concept of "Rewriting an equation," cosine similarity tells us that if the model thinks these concepts are logically related, even if one is more prominent, or has greater magnitude.

Further, cosine similarity should make an excellent tool for Variable Renaming, since the angle of the logic vector should remain nearly identical if the model understands the underlying proof structure.
## Machine Learning
### Linear Regression

In a standard linear regression, we assume a relationship between the independent variables $X$ and a dependent variable $y$. For our probe,
- **Input** ($X$): The activation vector from the residual stream. This is a vector is $\mathbb{R}^{768}$.
- **Output** ($y$): The property labeled, like Variable Count. This is a real scalar.
- **The Model**: We seek a weight vector $w\in\mathbb{R}^{768}$ and a bias $b$ such that $\hat{y}=w\cdot X+b$.

To find the best $w$, we minimize the Mean Squared Error (MSE). It measures the average squared difference between the model's prediction and the actual label:

$J(w,b)=\frac{1}{n}\sum_{i=1}^{n}\left(y_i-(w\cdot X_i+b)\right)^2

By minimizing this loss, we are rotating and scaling $w$ until it points in the direction that most closely aligns with how the label in question changes within the 768-dimensional space.

### Logistic Regression

To answer a binary question, we take our linear formula $z=w\cdot X+b$ and wrap it in a Sigmoid Function: $\sigma(z)=\frac{1}{1+e^{-z}}$. This forces the output to stay strictly between 0 and 1, which lets us interpret the result as a probability.

Logistic Regression find a hyperplane that splits the high-dimensional activation space in two. For example: on one side of the plan, the model predicts that there is no logical and symbol, but on the other side it does. The weight vector $w$ is normal to this plane. It is pointing directly in the "Presence of Logical And" direction in the model.  

Logistic Regression uses Binary Cross-Entropy Loss. We care not about Euclidean Distance, but certainty. For a single data point, BCEL is calculated as $L=-[y\log(\hat{y})+(1-y)\log(1-\hat{y})]$.

### Train/Test splitting for proofs

The samples from Lean proofs are highly correlated. So, instead of a random shuffle of all tactic states, we should collect all unique Theorem Names, randomly assign 80% to training and 20% to testing. 

### PCA & t-SNE & UMAP

#### Principal Component Analysis

Finds the directions that capture the most variance by taking the eigenvectors of a covariance matrix representing the data. If PCA clearly seperates two components, then we have evidence for global linear structure. Essentially, Principal Component Analysis is an application of the Spectral Theorem to the covariance matrix of activations. Given a centered data matrix $X\in\mathbb{R}^{n\times 768}$, with $n$ the number of Lean states, we compute the sample covariance matrix: $\Sigma=\frac{1}{n-1}X^TX$. Since $\Sigma$ is symmetric and positive semi-definite, it has an orthogonal eigenbasis. PCA finds the unit vector $w$ that maximizes the variance of the projected data:

$w_1=\argmax_{\Lvert w \rVert =1}\text{Var}(Xw)=\argmax_{\Lvert w \Rvert =1} w^T \Sigma w$. The solution $w_1$ is the eigenvector corresponding to the largest eigenvalue.

#### t-distributed Stochastic Neighbor Embedding

t-SNE is a non-linear probabilistic technique that looks for neighborhoods instead of variance. It tries to keep points that were close togther in 768-dimensional space close in 2 dimensions. It doesn't care about total or global distance. It's famous for forming beautiful clusters of ideas. t-SNE treats dimensionality reduction as an optimization problem where we minimze the Kullback-Leibler Divergence betwwen two probability distributions. We first look at a high-dimensional space, $P$: we calculate the similarity of a point $x_i$ to $x_j$ as a conditional probability of $j$ given $i$ using a Guassian distribution. This represents the likelihood of $x_i$ picking $x_j$ as its neighbor. Then, a low-dimensional space $Q$, we define similarity $q_{ij}$ between the mapped points $y_i$ and $y_j$ using the student's t-distribution with one degree of freedom. This heavy-tailed distribution helps with the crowding problem. 

We minimize the mismatch between $P$ and $Q$ as: $C=KL(P||Q)=\sum_i \sum_j p_{ij}\log\frac{p_{ij}}{q_{ij}}$. We use gradient descent to move the $y$ points around in 2D until the probability of the neighbors in the map matches the probability of the neighbords in the original 768D space.

#### Uniform Manifold Approximation and Projection

UMAP tries to do the same thing as t-SNE but also preserves global structure. It relies on Simplicial Complexes and Fuzzy Sets. We begin with Manifold Approximation: UMAP assumes the activations lie on a locally connected Riemann manifold. To find the manifold, it constructs a Vietoris-Rips complex by drawing a radius around every point and connecting them if they overlap. Then, to account for uneven data density, UMAP uses a fuzzy simplice, or radius, that grows and shrinks so that every point is connected to at least $k$ of its nearest neighbors. The result is a weighted graph where the edge represents the probability that a connection exists.

Finally, UMAP finds a low-dimensional representation that has the most similar fuzzy topological structure. It minimzes a specific form of Cross-Entropy:

$\sum_{e\in E}w_{H}(e)\log\frac{w_{H}(e)}{w_{L}(e)}+(1-w_{H}(e))\log\frac{1-w_{H}(e)}{1-w_{L}(e)}$, where $w_H$ is the edge weight in high dimensions and $w_L$ is the edge weight in low dimensions.

## Transformer Architecture
### The Residual Stream

In a classic feed-forward network, information is transformed layer by layer, until the original input is completely unrecognizable. In a Transformer, the architecture is built around Residual Connections, or skip connections. The state of the model at layer $n$ is not just the output of a function-- it is the sum of the previous state and an update. If $x_n$ is the vector in the stream at layer $n$, the next state is $x_{n+1}=x_n+\text{Sublayer}(x_n)$, where `Sublayer` is either a Multi-Head Attention Block or an MLP. Because of this $x+f(x)$ structure, the stream is a continuous vector space of dimension $d_{model}$ that runs throuogh the entire depth of the model.

The residual stream is essentially a running tally. At Layer 0 the stream is initialized with the embedding of the input tokens. As the vector travels through the layers attention heads read from the stream to see what other tokens are releveant, then write their findings back to the stream, and MLP's read the current state to perform logical or factual lookups, then write the result back. Since the operation is additive, the original information is recoverable. Hence, you can train a Linear probe at any layer.

The residual stream is a vector $v \in \mathbb{R}^{d_{model}}$. Each sublayer provides a displacement vector $\Delta v$. The stream is inherently linear because it is an additive, and therefore linear, combination of updates: $x_L=x_0+\sum_{i=0}^{L-1}\Delta v_i$. When training the probe, we look for a direction $w$ such that the projection $x_i\cdot w$ correlates with the label. If the model figures out that there are 3 variables in Layer 4, some sublayer in Layer 4 must have written a vector worth three variables into the stream.

The residual stream is where the reasoning lives. Layers 1 and 2 might just contain raw syntax, where Layers 6 through 8 might contain geometric representaions, and Layers 11 and 12 contain the logits.

### Layer-wise Evolution

The study of how a model's internal representation of a concept is incrementally constructed, refined, and predicted. It is a discrete dynamical system where the state vector $x$ moves through a 768-dimensional manifold guided by the vector fields defined by the attention and MLP layers. If we treat the residual stream as a vector space, every layers adds a displacement vector $\Delta x_n$. The evolution of the representation is the trajectory $x_0,x_1,\dots,x_L$. The Feature Extractors in the early layers mainly focus on de-noising and syntax. Probes usually fail on abstract concepts but will succeed on positional ones. The Semantic Convergers in the middle layers see the model combining syntactic tokens into abstract concepts. This is where the linear separability of labels peaks. The Logit Preprocessers in the late layers show the model begin to collapse the abstract representation into the specific tokens it needs to output next. Geometric clarity might actually decrease here.

To quantify this evolution, you train a battery of probes with one for every layer. Then, you plot Probe Accuracy (Loss) vs Layer Number. You can test if an evolution is causal using a technique called activation steering. We can take our direction $w$ and nudge the evolution: $x_6^{new}=x_6^{old}+\alpha w$. By adding this vector during the forward pass, you are manually making the model believe there are more variables than actually exist. If the model's predicted tactic changes to something that expects more variables, then that's causal evidence of layer-wise evolution leading to a specific concept driving the model's behavior.
## Mechanistic Interpretability
### Linear Representation Hypothesis

The claim that Large Language Models organize high-level concepts as directions, vectors, in a high-dimensional space. The LRH posits that the model maps a semantic concept $C$ to a specific vector $v_C$ in the residual stream. Any activation vector $x$ can be thought of as a sum of these concept vectors: $x=\sum_i \alpha_i v_C_i+\text{noise}$, where $\alpha_i$ is the intensity or presence of that concept. Since these concepts are stored as directions, we can use linear probes to slice the space and find them. If the representation were non-linear, a linear probe fails to find the cconcept, even if the model knew it.

Computation through projection is the leading theory for why a model why choose to store things linearly. When an attention head or an MLP layer reads the residual stream, it performs a linear transformation via a matrix multiplcation. If a concept is stored linearly, a single matrix operation can extract it via dot product: $\text{Activation}=\sigma(W\cdot x+b)$. By storing concepts as directions, the model makes it mathematically easy for subsequent layers to retrieve and use that information.
