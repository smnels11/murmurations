"""
Murmurations of elliptic curves: data acquisition from LMFDB, reproduction
of He-Lee-Oliver-Pozdnyakov's results, and PyTorch extensions.
"""

from .data import (
    AP_PREFIX,
    build_balanced_dataset,
    build_dataset,
    feature_matrix,
    load_or_build,
)
from .features import (
    average_ap_by_group,
    average_ap_by_rank,
    hasse_matrix,
    moment_channels,
    moment_feature_matrix,
    normalized_ap,
    primes_for_indices,
)
from .frobenius import ap_at_primes, ap_vector
from .lmfdb import connect, count_curves, fetch_curves
from .ml import fit_gr, logistic_experiment, pca_2d, repeat_logistic
from .primes import first_primes, prime
from .synthetic import make_synthetic_dataset
from .torch_models import (
    MLP,
    ApCNN,
    get_device,
    make_loaders,
    make_loaders_from_arrays,
    train,
)

__all__ = [
    # primes
    "first_primes",
    "prime",
    # frobenius
    "ap_vector",
    "ap_at_primes",
    # lmfdb
    "connect",
    "fetch_curves",
    "count_curves",
    # data
    "AP_PREFIX",
    "build_dataset",
    "build_balanced_dataset",
    "load_or_build",
    "feature_matrix",
    # features
    "average_ap_by_rank",
    "average_ap_by_group",
    "normalized_ap",
    "primes_for_indices",
    "hasse_matrix",
    "moment_feature_matrix",
    "moment_channels",
    # synthetic
    "make_synthetic_dataset",
    # ml
    "logistic_experiment",
    "repeat_logistic",
    "pca_2d",
    "fit_gr",
    # torch
    "get_device",
    "make_loaders",
    "make_loaders_from_arrays",
    "MLP",
    "ApCNN",
    "train",
]
