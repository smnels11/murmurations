"""
Classical ML reproductions: logistic regression, PCA, and curve fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import feature_matrix


def _balanced_sample(
    df: pd.DataFrame, ranks, n_per_rank: int, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for r in ranks:
        sub = df[df["rank"] == r]
        take = min(n_per_rank, len(sub))
        idx = rng.choice(sub.index, size=take, replace=False)
        parts.append(df.loc[idx])
    return pd.concat(parts).sample(frac=1.0, random_state=seed)


@dataclass
class ClassifierResult:
    ranks: tuple
    n_features: int
    accuracy: float
    mcc: float  # Matthews correlation coefficient == paper's "confidence"
    model: LogisticRegression
    scaler: StandardScaler


def logistic_experiment(
    df: pd.DataFrame,
    ranks,
    n_features: int = 1000,
    n_per_rank: int | None = None,
    test_size: float = 0.25,
    standardize: bool = True,
    seed: int = 0,
    max_iter: int = 2000,
    featurizer=None,
) -> ClassifierResult:
    """
    Reproduce the paper's logistic-regression rank classification (Table 1).

    Binary (len(ranks)==2) or multinomial (>2). Reports test accuracy and the
    Matthews correlation coefficient, which the paper labels "confidence".

    featurizer : optional callable (data_df, num_primes) -> X. Defaults to raw
        a_p (feature_matrix). Pass features.moment_feature_matrix to hand the
        linear model the Hasse-normalized moment features (mean + skew signal).
    """
    if n_per_rank is None:
        # Min class size among the REQUESTED ranks only. (Grouping the boolean
        # .isin() mask by rank counts every rank present in df, so a scarce rank
        # sitting in the frame -- e.g. 531 rank-3 curves -- would silently cap
        # every experiment, including ones that never asked for it.)
        n_per_rank = int(
            df[df["rank"].isin(ranks)].groupby("rank").size().min()
        )
    data = _balanced_sample(df, ranks, n_per_rank, seed=seed)

    featurizer = featurizer or feature_matrix
    X = featurizer(data, num_primes=n_features)
    y = data["rank"].to_numpy()

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler().fit(Xtr) if standardize else None
    if scaler is not None:
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    return ClassifierResult(
        ranks=tuple(ranks),
        n_features=n_features,
        accuracy=float(accuracy_score(yte, pred)),
        mcc=float(matthews_corrcoef(yte, pred)),
        model=clf,
        scaler=scaler,
    )


@dataclass
class PCAResult:
    scores: np.ndarray  # (n_curves, 2): PC1, PC2
    ranks: np.ndarray
    components: np.ndarray  # (2, n_features): rows are PC1, PC2 weights
    explained_variance_ratio: np.ndarray
    pca: PCA


def pca_2d(
    df: pd.DataFrame,
    ranks=None,
    n_features: int = 1000,
    standardize: bool = True,
    n_per_rank: int | None = None,
    seed: int = 0,
) -> PCAResult:
    """
    Project v_L(E) into R^2 with PCA (Figs 2-5). ``components`` gives the
    a_p weights that make up PC1/PC2.

    n_per_rank : if given, take a balanced random sample of this many curves per
                 rank first. PCA is UNSUPERVISED -- it maximizes variance and
                 knows nothing about ranks -- so class proportions directly
                 shape the components. The paper's Fig 2 uses 12,000 per rank;
                 leaving the natural (heavily rank-0/1 skewed) proportions makes
                 PC1 lock onto the rank-0-vs-1 murmuration axis and rank 2 will
                 not separate along it.
    """
    data = df if ranks is None else df[df["rank"].isin(ranks)]
    if n_per_rank is not None:
        rs = ranks if ranks is not None else sorted(data["rank"].unique())
        data = _balanced_sample(data, rs, n_per_rank, seed=seed)
    X = feature_matrix(data, num_primes=n_features)
    if standardize:
        X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)
    return PCAResult(
        scores=scores,
        ranks=data["rank"].to_numpy(),
        components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        pca=pca,
    )


# --- Curve fitting for g_r(p): y = A x^alpha sin(B x^beta) (Eq. 4.10) ---


def _model(x, A, alpha, B, beta):
    return A * np.power(x, alpha) * np.sin(B * np.power(x, beta))


@dataclass
class FitResult:
    params: tuple  # (A, alpha, B, beta)
    mse: float

    def __call__(self, x):
        return _model(np.asarray(x, dtype=float), *self.params)


def fit_gr(
    p: np.ndarray, g: np.ndarray, p0=(0.5, 0.2, 0.1, 0.5), maxfev: int = 20000
) -> FitResult:
    """
    Fit y = A x^alpha sin(B x^beta) to (p, g_r(p)) by least squares (Table 2).

    Both signs of the initial A are tried and the lower-MSE fit kept. The
    model has a degenerate basin at B -> 0 (there sin(B x^beta) ~ B x^beta,
    a pure power law identifying only the product A*B), and an initial A of
    the wrong sign -- e.g. the default +0.5 against a rank-1 target, whose
    mirror-image oscillation needs A < 0 -- reliably sends the optimizer
    into it rather than across the A = 0 ridge.
    """
    p = np.asarray(p, dtype=float)
    g = np.asarray(g, dtype=float)
    best = None
    for sign in (1.0, -1.0):
        signed_p0 = (sign * p0[0], *p0[1:])
        try:
            popt, _ = curve_fit(_model, p, g, p0=signed_p0, maxfev=maxfev)
        except RuntimeError:
            continue
        mse = float(np.mean((_model(p, *popt) - g) ** 2))
        if best is None or mse < best.mse:
            best = FitResult(params=tuple(popt), mse=mse)
    if best is None:
        raise RuntimeError("curve_fit failed to converge from either sign")
    return best


def repeat_logistic(df, ranks, seeds=None, **kwargs) -> dict:
    """
    Run logistic_experiment over several seeds; return mean/std accuracy & mcc.
    """
    if seeds is None:
        seeds = range(5)
    accs, mccs = [], []
    for s in seeds:
        r = logistic_experiment(df, ranks, seed=s, **kwargs)
        accs.append(r.accuracy)
        mccs.append(r.mcc)
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "mcc_mean": float(np.mean(mccs)),
        "mcc_std": float(np.std(mccs)),
        "accs": accs,
    }
