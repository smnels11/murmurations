"""
The averaged quantities at the heart of the paper.

f_r(n) (Eq. 1.1 / 4.7): mean of a_{p_n}(E) over curves E of rank r in a fixed
conductor range. Plotting (n, f_r(n)) produces the murmuration.

g_r(p) (Eq. 4.9): the same averages viewed as a function of the prime p_n rather
than the index n -- only a log-scale reparametrization, but the natural x-axis
for curve fitting.

tilde a_p = a_p / (2 sqrt p) (Eq. 4.8): the Hasse-normalized coefficient in
[-1, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import AP_PREFIX
from .primes import first_primes


def _ap_cols(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith(AP_PREFIX))


def primes_for_indices(num_primes: int) -> np.ndarray:
    """Array [p_1, ..., p_num_primes]."""
    return np.array(first_primes(num_primes), dtype=np.int64)


def average_ap_by_group(
    df: pd.DataFrame, by="rank", groups=None, num_primes: int | None = None
) -> pd.DataFrame:
    """
    Mean of a_{p_n} within each group -- the paper's f_r(n) generalised.

    by : column name, or a same-length array of labels. Use 'rank' for the
        paper's murmuration; pass torsion labels to ask whether a *torsion*
        murmuration exists (unexplored in the paper).

    Returns a frame indexed by n = 1..num_primes with column 'p' (the prime)
    and one column per group holding that group's mean a_{p_n}.
    """
    cols = _ap_cols(df)
    if num_primes is not None:
        cols = cols[:num_primes]
    n_primes = len(cols)

    key = df[by].to_numpy() if isinstance(by, str) else np.asarray(by)
    if groups is None:
        groups = sorted(pd.unique(key))

    out = pd.DataFrame(index=pd.RangeIndex(1, n_primes + 1, name="n"))
    out["p"] = primes_for_indices(n_primes)
    for g in groups:
        out[str(g)] = df.loc[key == g, cols].mean(axis=0).to_numpy()
    return out


def average_ap_by_rank(
    df: pd.DataFrame, ranks=None, num_primes: int | None = None
) -> pd.DataFrame:
    """
    Compute f_r(n) for each requested rank (Eq. 1.1 / 4.7).
    """
    return average_ap_by_group(
        df, by="rank", groups=ranks, num_primes=num_primes
    )


def hasse_matrix(df: pd.DataFrame, num_primes: int | None = None) -> np.ndarray:
    """
    The Hasse-normalized feature matrix: column n is a_{p_n} / (2 sqrt p_n).

    Bounded to [-1, 1] and homoscedastic across primes (raw a_p has std ~
    sqrt p), which is much friendlier to both distance-based and gradient-based
    models.
    """
    cols = _ap_cols(df)
    if num_primes is not None:
        cols = cols[:num_primes]
    A = df[cols].to_numpy(dtype=np.float64)
    p = primes_for_indices(len(cols)).astype(np.float64)
    return A / (2.0 * np.sqrt(p))[None, :]


def moment_feature_matrix(
    df: pd.DataFrame, num_primes: int | None = None, moments=(1, 2, 3)
) -> np.ndarray:
    """
    Concatenate powers of the Hasse-normalized coefficients.

    moments=(1,) is just tilde a_p (odd -> carries mean/murmuration signal).
    Adding 2 (even -> spread) and 3 (odd -> SKEW) lets a *linear* model read the
    third-moment signal that separates rank 0 from rank 1 where the means
    coincide.
    """
    T = hasse_matrix(df, num_primes)
    blocks = [np.power(T, m) for m in moments]
    return np.concatenate(blocks, axis=1)


def moment_channels(
    df: pd.DataFrame, num_primes: int | None = None, moments=(1, 2, 3)
) -> np.ndarray:
    """
    Stack powers of tilde a_p as CHANNELS for Conv1d: shape (N, C, P).

    Same information as moment_feature_matrix, but laid out so a 1-D CNN can
    convolve along the prime axis with the moments as channels. Channel m at
    position n is (a_{p_n} / 2 sqrt p_n)^m.

    Why channels rather than one long vector: the discriminative signal
    oscillates *smoothly* in n (Fig 5), so neighbouring primes carry related
    information. A convolution pools them; a dense layer would have to learn
    that adjacency from scratch.
    """
    T = hasse_matrix(df, num_primes)  # (N, P)
    return np.stack([np.power(T, m) for m in moments], axis=1)  # (N, C, P)


def normalized_ap(df: pd.DataFrame, prime_index: int) -> pd.Series:
    """
    tilde a_{p} = a_p / (2 sqrt p) for the given 1-indexed prime_index.

    Returned per-curve, aligned with df's index, carrying the curve's rank via
    a companion column is left to the caller (use df['rank']).
    """
    p = float(first_primes(prime_index)[-1])
    col = f"{AP_PREFIX}{prime_index:04d}"
    return df[col].to_numpy() / (2.0 * np.sqrt(p))
