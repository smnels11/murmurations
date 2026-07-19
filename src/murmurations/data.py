"""
Build and cache datasets: LMFDB metadata + computed a_p feature vectors.

A dataset is a DataFrame with metadata columns lmfdb_label, lmfdb_iso,
conductor, rank plus a_p feature columns named ap0001 ... ap{num_primes:04d},
where column ap{n:04d} holds a_{p_n}(E). The a_p matrix (v_L(E) in the paper)
is recovered with ``feature_matrix``.

Datasets are cached to parquet so the (slow) LMFDB pull + a_p computation
only happens once.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .frobenius import ap_vector
from .lmfdb import fetch_curves

AP_PREFIX = "ap"


def _ap_columns(num_primes: int) -> list[str]:
    return [f"{AP_PREFIX}{n:04d}" for n in range(1, num_primes + 1)]


def build_dataset(
    conductor_min: int,
    conductor_max: int,
    ranks: Sequence[int] | None = None,
    num_primes: int = 1000,
    one_per_class: bool = True,
    limit: int | None = None,
    random_sample: bool = False,
    seed: float | None = None,
    progress: bool = True,
    desc: str = "computing a_p",
) -> pd.DataFrame:
    """
    Fetch curves and compute their a_p vectors into a single DataFrame.
    """
    meta = fetch_curves(
        conductor_min,
        conductor_max,
        ranks=ranks,
        one_per_class=one_per_class,
        limit=limit,
        random_sample=random_sample,
        seed=seed,
    )

    # Preallocate an int32 array and fill row-by-row.
    n = len(meta)
    ap_arr = np.empty((n, num_primes), dtype=np.int32)

    ainvs_iter = enumerate(meta["ainvs"])
    if progress:
        ainvs_iter = tqdm(ainvs_iter, total=n, desc=desc)

    for i, ainvs in ainvs_iter:
        ap_arr[i, :] = ap_vector(ainvs, num_primes)

    ap_df = pd.DataFrame(ap_arr, columns=_ap_columns(num_primes))

    out = pd.concat(
        [meta.drop(columns=["ainvs"]).reset_index(drop=True), ap_df],
        axis=1,
        copy=False,
    )
    return out


def build_balanced_dataset(
    conductor_min: int,
    conductor_max: int,
    ranks: Sequence[int] = (0, 1, 2),
    n_per_rank: int = 20_000,
    num_primes: int = 1000,
    seed: float = 0.42,  # Postgres setseed() arg, float in [-1, 1]
    progress: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Balanced random sample of ``n_per_rank`` curves per rank.

    This is what the paper actually does for Table 1 (20,000 curves per rank,
    conductor in [1, 1e5]) -- NOT every curve in the range, which is ~330k
    isogeny classes and will exhaust a 16 GB machine.

    Ranks with fewer than n_per_rank available simply return what exists (e.g.
    there are only 531 rank-3 curves below conductor 1e5).
    """
    frames = []
    for r in ranks:
        frames.append(
            build_dataset(
                conductor_min,
                conductor_max,
                ranks=[r],
                num_primes=num_primes,
                limit=n_per_rank,
                random_sample=True,
                seed=seed,
                progress=progress,
                desc=f"a_p rank {r}",
                **kwargs,
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_or_build(
    cache_path: str | Path, builder=None, **kwargs
) -> pd.DataFrame:
    """
    Load a cached parquet dataset, or build (and cache) it if absent.

    builder : callable returning the DataFrame.
        Defaults to :func:`build_dataset`;
        pass e.g. partial(build_balanced_dataset, ...) for per-rank sampling.
    kwargs are forwarded to the builder.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    df = (builder or build_dataset)(**kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def feature_matrix(
    df: pd.DataFrame, num_primes: int | None = None
) -> np.ndarray:
    """Return the a_p matrix X of shape (n_curves, num_primes) as float64.

    If num_primes is given, only the first that many a_p columns are used (this
    is how the paper runs experiments in lower dimension d).
    """
    cols = [c for c in df.columns if c.startswith(AP_PREFIX)]
    cols = sorted(cols)
    if num_primes is not None:
        cols = cols[:num_primes]
    return df[cols].to_numpy(dtype=np.float64)
