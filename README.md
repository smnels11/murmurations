# Murmurations of Elliptic Curves

This repository reproduces selected results from [*Murmurations of Elliptic Curves*](https://arxiv.org/abs/2204.10140) by Y.H. He, K.H. Lee, T. Oliver, and A. Pozdnyakov, hereafter abbreviated as **HLOP**, while also building on their work. At a high level, the work of HLOP applies machine learning techniques to datasets in which elliptic curves are treated as observations and different collections of their numerical invariants serve as features.

The term *murmuration* refers to the striking oscillatory patterns that emerge when visualizing certain arithmetic invariants of elliptic curves. The name is inspired by the natural phenomenon of *bird murmurations*, in which thousands of birds fly in dense, coordinated formations whose continually shifting shapes resemble the patterns observed in the data.

The original HLOP paper linked above provides a thorough explanation of the mathematics underlying this project. Readers interested in the broader context are encouraged to consult the authors' related publications cited therein. For those with a basic understanding of number theory, elliptic curves, and machine learning, see T. Oliver's excellent talk at the Isaac Newton Institute for Mathematical Sciences' workshop on [Number Theory, Machine Learning, and Quantum Black Holes](https://www.newton.ac.uk/event/blhw01/), which is available on the [institute's website](https://www.newton.ac.uk/seminar/39370/) and [YouTube](https://www.youtube.com/watch?v=AgmMQyhVK4A&t=771s).

This README provides a roadmap of the repository and an explanation of the accompanying code, followed by a concise overview of the mathematical objects of interest.

## Results at a Glance

**Reproductions** (all numbers from the committed notebook outputs):

- **Rank classification, HLOP Table 1** (logistic regression on 1000 Frobenius traces; conductor $[1, 10^5]$, 20,000 curves per rank):

  | Ranks     | HLOP acc | Ours  | HLOP MCC | Ours |
  |-----------|----------|-------|----------|------|
  | {0, 1}    | 0.961    | 0.960 | 0.92     | 0.92 |
  | {0, 2}    | 0.996    | 0.993 | 0.99     | 0.99 |
  | {1, 2}    | 0.999    | 0.999 | 0.99     | 1.00 |
  | {0, 1, 2} | 0.975    | 0.974 | 0.96     | 0.96 |

- **Section 5 heuristic** (rank ≤ 1 vs rank ≥ 2 from only the first ten $a_p$): 0.970 on conductor $[1, 10^4]$, matching the paper's ≈ 0.97. Widening the window to $[1, 10^5]$ drops accuracy to 0.894 — the per-prime statistics are conductor-dependent, so wide windows mix differently-shaped distributions (Notebook 03).
- **Section 4.5 curve fits**: both $g_{0}(p)$ and $g_{1}(p)$ fit $A\,x^{\alpha}\sin(B\,x^{\beta})$ with $\beta = 0.54$ and mirrored amplitudes, matching Table 2's universal $\beta \approx 0.5$ (Notebook 02).

**Findings** (beyond the paper):

- **The murmuration mean carries essentially all the linearly accessible rank signal.** A synthetic testbed with independently switchable mean and skew signals shows a linear model on raw $a_p$ is blind to the third-moment (skew) signal while explicit moment features recover it. However, on real curves the moment features add nothing (0.977 vs 0.987 for rank 0 vs 1), at every feature count tried (Notebook 05).
- **The MLP's bottleneck is optimization, not representation.** It can represent $x \mapsto x^3$, yet never finds the skew signal from raw inputs at any capacity tried (Notebook 05) — while a single-channel 1-D CNN finds that same signal (~0.97) through weight sharing alone (Notebook 06).
- **Locality is real on real curves, but redundant.** The CNN loses to the linear baseline (0.915 ± 0.005 vs 0.987); a pre-registered prime-shuffle control then collapses it to 0.721 ± 0.012 while logistic regression is unchanged — the CNN genuinely uses the sequence structure, but the ordering information is already contained in what the linear model extracts (Notebook 06).


## Getting Started

This project uses [`uv`](https://docs.astral.sh/uv/) for Python package and environment management.

After cloning the repository, install the project dependencies with:

```bash
uv sync
```

This will create a local virtual environment (`.venv/`) and install all dependencies specified in `pyproject.toml` and locked in `uv.lock`.

### Development

Code style is enforced with [`ruff`](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`): 
```bash
uv run ruff check . && uv run ruff format --check .
```

### Data access

Elliptic curve metadata is pulled from the [LMFDB read-only SQL mirror](https://www.lmfdb.org/api/options), a public PostgreSQL instance maintained by the LMFDB collaboration. The connection settings, including the username and password, are **public, read-only credentials published by LMFDB** at the link above. Since they are documented public constants rather than secrets, they are defined directly in `src/murmurations/lmfdb.py`; no account, API key, or configuration is required.


**Note:** This project was developed on Apple Silicon and uses PyTorch's `mps` backend when available, which is supported through the standard PyPI distribution. For CUDA-enabled systems, PyTorch should be installed using the appropriate CUDA wheel index. See the official `uv` documentation on [Using a PyTorch Index](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index) for configuring GPU-enabled PyTorch installations.

At runtime, the package automatically selects the best available device:
`mps` $\rightarrow$ `cuda` $\rightarrow$ `cpu`
The device selection logic is centralized in `get_device()` within `src/murmurations/torch_models.py`.


## Repository Overview

The repository is organized as the following computational pipeline:

1. Query elliptic curve metadata from the [LMFDB Postgres Mirror](https://www.lmfdb.org/api/options).
2. Compute and cache Frobenius trace vectors for the first 1000 primes using [PARI/GP](https://pari.math.u-bordeaux.fr/) via [`cypari2`](https://pypi.org/project/cypari2/).
3. Construct machine learning datasets from the cached arithmetic data.
4. Reproduce the selected experiments and visualizations of HLOP using `scikit-learn` and `matplotlib`.
5. Extend HLOP's work by exploring convolutional neural network architectures.

The primary entry point is the `notebooks/` directory, read in order:

| Notebook                    | Contents                                                                                                                 |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `01_data_acquisition`       | LMFDB pull, $a_p$ computation, parquet caching; sanity-checks curve counts against the paper's Example 1.                |
| `02_murmurations`           | The phenomenon itself: Figs 6–7, Hasse-normalized histograms, and the Section 4.5 curve fits.                            |
| `03_classification_sklearn` | Table 1 reproduction, PCA (Figs 2, 5), and the Section 5 low-dimensional heuristic with its conductor-window dependence. |
| `04_pytorch_dnn`            | A first deep model: MLP vs the logistic baseline on rank 0 vs 2 (depth buys essentially nothing).                        |
| `05_rank`                   | Which signal carries the information? Synthetic mean/skew testbed, then the real-data verdict.                           |
| `06_cnn_sequence`           | The sequence model: 1-D CNN and the prime-shuffle control that decomposes weight sharing from locality.                  |

All notebooks are committed with outputs - no LMFDB access or MPS/GPU is needed to review them.

### Repository Structure

```text
murmurations/
├── data/
├── notebooks/
├── src/
│   └── murmurations/
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock
```

### `src/murmurations/`

The `murmurations` package contains the reusable Python functionality underlying the project. Modules are organized by responsibility, separating data acquisition, arithmetic computations, feature engineering, machine learning experiments, and visualization.

```text
data.py          Build, cache, and load datasets.
features.py      Feature engineering and tensor construction.
frobenius.py     Compute Frobenius trace vectors using PARI/GP.
lmfdb.py         Query the LMFDB PostgreSQL mirror.
ml.py            Classical machine learning experiments.
plots.py         Visualization utilities for arithmetic data and model results.
primes.py        Prime utilities and cached prime generation.
synthetic.py     Synthetic datasets for controlled experiments.
torch_models.py  PyTorch datasets, models, and training loops.
```

### Module Dependency Graph

The `murmurations` package is organized as a directed acyclic graph of imports. Reading the modules in topological order follows the lifecycle of the data pipeline:

```text
                LMFDB
                  │
                  ▼
primes.py      lmfdb.py
    │             │
    ▼             ▼
frobenius.py ─► data.py
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
 features.py  synthetic.py  ml.py
```

Arrows point from imported to importer. Three edges are omitted from the picture for legibility: `primes.py` is also imported directly by `features.py`, `synthetic.py`, and `plots.py` (prime lookup tables).

`data.py` is the central integration point: everything above it acquires and transforms arithmetic data; everything below it consumes the resulting datasets. Two modules sit outside the graph entirely — `plots.py` and `torch_models.py` import nothing from the acquisition or dataset layers by design. They operate on plain DataFrames and NumPy arrays, which is what lets Notebooks 05–06 feed them synthetic and real data interchangeably.

## Mathematical Theory

For the remainder of this section, let:
- $E$ be an elliptic curve over $\mathbb{Q}$; and
- $p$ be a prime.

### The Conductor of an Elliptic Curve

The *conductor* of $E$, denoted $N(E)$, is

$$
N(E) := \prod_{p \text{ prime}} p^{e_{p}} ,
$$

where

$$
e_{p} := \begin{cases}
0              & \text{ if } E \text{ has good reduction at } p , \\
1              & \text{ if } E \text{ has multiplicative reduction at } p , \\
2 + \delta_{p} & \text{ if } E \text{ has additive reduction at } p . \\
\end{cases}
$$

We will not discuss $\delta_{p}$ beyond noting that it is a measure of the wild ramification in the action of the inertia group on the Tate module $T_{\ell}(E)$.

For our purposes, the following properties are the most relevant:
* $N(E)$ encodes the primes for which $E$ has bad reduction; and 
* $N(E)$ is invariant under isogeny.

Throughout this project, we use the conductor to organize and construct datasets of elliptic curves. For example, one may consider all curves whose conductors lie within prescribed bounds. To this end, for positive integers $N_{1}, N_{2}$, we define the set $\mathcal{E}[N_{1}, N_2]$ to be the set of elliptic curves over $\mathbb{Q}$ satisfying $N_{1} \leq N(E) \leq N_{2}$:

```math
\mathcal{E}[N_{1}, N_{2}] := \{ E / \mathbb{Q} : N_{1} \leq N(E) \leq N_{2} \} .
```

In practice, we consider *isogeny classes* of elliptic curves in $\mathcal{E}[N_{1}, N_{2}]$ since $N(E)$ is invariant under isogeny.

### $L$-Functions and Frobenius Traces

If $E$ has good reduction at $p$, $L_{p}(E, s)$ is the *local* $L$-*factor* defined by

$$
L_{p}(E, s) := 1 - a_{p}(E)p^{-s} + p^{1 - 2s} ,
$$

where $s \in \mathbb{C}$,

$$
a_{p}(E) := p + 1 - \lvert E(\mathbb{F}_{p}) \rvert,
$$

and $`\lvert \mathbb{F}_{p}(E) \rvert`$ is the number of points of $E$ over $\mathbb{F}_{p}$.

For primes $p$ of bad reduction, $L_{p}(E, s)$ is

$$
L_{p}(E, s) := 1 - a_{p}(E)p^{-s} ,
$$

where

$$
a_{p}(E) := \begin{cases}
 1 & \text{ if } E \text{ has split multiplicative reduction at } p , \\
-1 & \text{ if } E \text{ has nonsplit multiplicative reduction at } p , \\
 0 & \text{ if } E \text{ has additive reduction at } p .
\end{cases}
$$

The $L$-*function* of $E$ is defined by

$$
L(E, s) := \prod_{p \text{ prime}} \frac{1}{L_{p}(E, s)} .
$$

The product defining $L(E, s)$ converges in the half-plane $\Re(s) > \frac{3}{2}$. It is well known that $L(E, s)$ admits a holomorphic continuation to all of $\mathbb{C}$ and satisfies a functional equation relating $s$ and $2 - s$.

The quantities $a_{p}(E)$ are known as the *Frobenius traces* of $E$ and, like $N(E)$, are isogeny-invariant. They are central to many important questions in number theory, but also play an important role in HLOP's work. In the machine learning framework described earlier, elliptic curves constitute the observations in the datasets of interest, while collections of Frobenius traces (and quantities derived from them) serve as the features used to train and analyze models.

### Rank and Torsion

By the Mordell-Weil Theorem (1922), the set of $\mathbb{Q}$-rational points of $E$, denoted $E(\mathbb{Q})$, forms a finitely generated Abelian group. Consequently, $E(\mathbb{Q})$ admits a decomposition of the form

$$
E(\mathbb{Q}) \simeq E(\mathbb{Q})_{\mathrm{tors}} \oplus \mathbb{Z}^{r_{E}} ,
$$

where $`E(\mathbb{Q})_{\mathrm{tors}}`$ is the finite subgroup consisting of points of finite order, called the *torsion subgroup*, and $r_{E} \in \mathbb{Z}_{\geq 0}$ is the *rank* of $E$.

The rank is one of the most mysterious arithmetic invariants of an elliptic curve. Despite decades of intensive study, no algorithm is known that provably computes the rank of every elliptic curve over $\mathbb{Q}$. Further, whether the set of ranks is bounded or not remains an open question.

The famed Birch-Swinnerton-Dyer (BSD) conjecture predicts the rank $r_{E}$ is equal to the order of vanishing of $L(E, s)$ at $s = 1$. Although this equality remains unproven in full generality, the work of Kolyvagin, together with the proof of the Modularity Theorem, establishes the BSD Conjecture holds for elliptic curves of rank 0 and 1. 

Moreover, it is conjectured, in a rigorous sense, that 50% of elliptic curves over $\mathbb{Q}$ have rank 0, 50% have rank 1, and 0% have higher rank. Consequently, many machine learning studies (including HLOP) frame rank prediction as a binary classification problem: distinguishing elliptic curves of rank 0 from those of positive rank. Several of the experiments in this repository follow this same paradigm.