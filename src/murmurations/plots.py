"""
Plotting helpers reproducing the paper's figures.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colour convention matching the paper: rank 0 blue, 1 red, 2 green, 3 orange.
RANK_COLORS = {0: "tab:blue", 1: "tab:red", 2: "tab:green", 3: "tab:orange"}


def plot_murmuration(
    favg: pd.DataFrame, ranks=None, x="n", s=6, ax=None, title=None
):
    """
    Scatter (n, f_r(n)) for each rank -- Figures 1, 6, 7, 8.

    favg is the output of features.average_ap_by_rank. x is 'n' (prime index) or
    'p' (the prime itself).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    xs = favg.index.to_numpy() if x == "n" else favg["p"].to_numpy()
    rank_cols = [c for c in favg.columns if c != "p"]
    if ranks is not None:
        rank_cols = [str(r) for r in ranks]
    for c in rank_cols:
        ax.scatter(
            xs,
            favg[c],
            s=s,
            color=RANK_COLORS.get(int(c), None),
            label=f"rank {c}",
        )
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("prime index n" if x == "n" else "prime p")
    ax.set_ylabel(r"average $a_p$")
    ax.legend(markerscale=2)
    if title:
        ax.set_title(title)
    return ax


def plot_pca(pca_result, ax=None, s=5, alpha=0.5, title=None):
    """
    PC1 vs PC2 scatter coloured by rank -- Figures 2, 4.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    scores, ranks = pca_result.scores, pca_result.ranks
    for r in np.unique(ranks):
        m = ranks == r
        ax.scatter(
            scores[m, 0],
            scores[m, 1],
            s=s,
            alpha=alpha,
            color=RANK_COLORS.get(int(r), None),
            label=f"rank {r}",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=3)
    if title:
        ax.set_title(title)
    return ax


def plot_pc1_weights(pca_result, ax=None, title=None):
    """
    The a_p weights making up PC1 -- Figures 3, 5 (the decaying oscillation).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(
        np.arange(1, pca_result.components.shape[1] + 1),
        pca_result.components[0],
        s=6,
        color="k",
    )
    ax.set_xlabel("vector index n")
    ax.set_ylabel(r"weight of $a_{p_n}$ in PC1")
    if title:
        ax.set_title(title)
    return ax


def plot_ap_histograms(
    df, prime_indices, rank, normalize_fn, bins=25, axes=None
):
    """
    Histograms of tilde a_p at several primes for one rank -- Figures 9-11.

    normalize_fn(df, prime_index) -> per-curve normalized values
    (features.normalized_ap).
    """
    sub = df[df["rank"] == rank]
    if axes is None:
        _, axes = plt.subplots(
            1, len(prime_indices), figsize=(3 * len(prime_indices), 3)
        )
    axes = np.atleast_1d(axes)
    from .primes import first_primes

    for ax, pi in zip(axes, prime_indices, strict=True):
        vals = normalize_fn(sub, pi)
        ax.hist(
            vals,
            bins=bins,
            range=(-1, 1),
            color=RANK_COLORS.get(int(rank), None),
            density=True,
        )
        ax.set_title(rf"$\tilde a_{{{first_primes(pi)[-1]}}}$")
    return axes


def plot_curve_fit(p, g, fit_result, color="tab:blue", ax=None, label=None):
    """
    Overlay the fitted A x^alpha sin(B x^beta) on g_r(p) -- Figures 13-16.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(p, g, s=5, color=color, alpha=0.4, label=label)
    xs = np.linspace(np.min(p), np.max(p), 2000)
    ax.plot(xs, fit_result(xs), color="orange", lw=2)
    ax.set_xlabel("prime p")
    ax.set_ylabel(r"average $a_p$")
    return ax
