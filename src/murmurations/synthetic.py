"""
A synthetic testbed that mimics the rank-0 vs rank-1 signal structure.

The real difficulty of rank 0 vs 1 is that, at each prime p, the two ranks'
distributions of the Hasse-normalized coefficient tilde a_p = a_p/(2 sqrt p)
overlap heavily. They differ in two ways:

    * MEAN: the murmuration -- f_0 and f_1 mirror each other, amplitude and
            period growing with p. A difference in class-conditional means is
            visible to a linear classifier on raw a_p.
    * SKEW: rank 0 slightly left-skewed, rank 1 slightly right-skewed, the
            skew itself oscillating in p. A third-moment signal; a linear
            model on raw a_p is blind to it where the means coincide.

Here we draw tilde a_p from a skew-normal with a prescribed mean and skew per
prime, then map back to integer a_p within the Hasse bound. This lets us switch
each signal on/off and see exactly which model exploits which -- ground truth we
never have with the real data.

Output is a DataFrame in the SAME format as data.build_dataset
(rank + ap0001..), so every downstream tool works on it unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import AP_PREFIX
from .primes import first_primes


def _skewnorm_samples(n, mean, skew_a, scale, rng):
    """
    Skew-normal samples with a *target mean* and shape parameter skew_a.
    """
    delta = skew_a / np.sqrt(1.0 + skew_a**2)
    loc = mean - scale * delta * np.sqrt(2.0 / np.pi)  # so E[X] == mean
    return stats.skewnorm.rvs(
        a=skew_a, loc=loc, scale=scale, size=n, random_state=rng
    )


def make_synthetic_dataset(
    n_per_rank: int = 4000,
    num_primes: int = 1000,
    mean_amp: float = 0.06,  # strength of the murmuration (mean) signal
    skew_amp: float = 4.0,  # strength of the skew signal (skew-normal shape)
    scale: float = 0.35,  # per-prime spread of tilde a_p
    conductor_range=(7500, 10000),
    seed: int = 0,
) -> pd.DataFrame:
    """
    Generate a rank-{0,1} synthetic dataset with tunable mean/skew signals.

    Set mean_amp=0 to isolate the skew signal (logistic-on-raw should collapse
    to ~chance while moment features / MLP still separate). Set skew_amp=0 to
    isolate the murmuration (logistic-on-raw should suffice).
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(first_primes(num_primes), dtype=float)
    n = np.arange(1, num_primes + 1)

    # Growing-amplitude, growing-period oscillation (as in the paper's fits).
    osc = np.sin(0.12 * np.sqrt(p)) * (p**0.2)
    mean0 = mean_amp * osc  # rank 0 mean of tilde a_p
    mean1 = -mean_amp * osc  # rank 1 mirrors it

    # Skew oscillates on a slower schedule; rank 1 is the mirror of rank 0.
    skew0 = skew_amp * np.sin(0.05 * np.sqrt(p))
    skew1 = -skew0

    two_sqrt_p = 2.0 * np.sqrt(p)

    def draw(mean_vec, skew_vec):
        cols = np.empty((n_per_rank, num_primes), dtype=np.int32)
        for j in range(num_primes):
            t = _skewnorm_samples(
                n_per_rank, mean_vec[j], skew_vec[j], scale, rng
            )
            t = np.clip(t, -1.0, 1.0)
            a = np.rint(t * two_sqrt_p[j])
            bound = np.floor(two_sqrt_p[j])
            cols[:, j] = np.clip(a, -bound, bound).astype(np.int32)
        return cols

    a0, a1 = draw(mean0, skew0), draw(mean1, skew1)
    colnames = [f"{AP_PREFIX}{i:04d}" for i in n]

    frames = []
    for rank, arr in [(0, a0), (1, a1)]:
        d = pd.DataFrame(arr, columns=colnames)
        d.insert(0, "rank", rank)
        d.insert(
            0,
            "conductor",
            rng.integers(conductor_range[0], conductor_range[1], n_per_rank),
        )
        d.insert(0, "lmfdb_iso", "synthetic")
        d.insert(0, "lmfdb_label", "synthetic")
        frames.append(d)

    return (
        pd.concat(frames, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
