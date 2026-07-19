"""

Frobenius traces a_p(E) computed from a-invariants via PARI (cypari2).

For a good prime p, a_p(E) = p + 1 - #E(F_p). For bad primes PARI's ellap
returns the correct a_p in {-1, 0, 1} depending on the reduction type, so the
same call works for every prime. This matches SageMath's E.ap(p) (Sage calls
PARI under the hood) and reproduces LMFDB's stored aplist exactly.

a_p is an isogeny-class invariant, so any representative of a class yields the
same vector -- consistent with the paper using one curve per isogeny class.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cypari2

from .primes import first_primes

_pari = cypari2.Pari()


def _ellinit(ainvs: Sequence[int]):
    # PARI accepts the 5-tuple [a1, a2, a3, a4, a6]. LMFDB stores the minimal
    # Weierstrass model, which is what we want for correct bad-prime a_p.
    return _pari.ellinit([int(a) for a in ainvs])


def ap_at_primes(ainvs: Sequence[int], primes: Iterable[int]) -> list[int]:
    """
    Compute a_p(E) at the given primes for a curve given by its a-invariants.
    """
    E = _ellinit(ainvs)
    return [int(_pari.ellap(E, int(p))) for p in primes]


def ap_vector(ainvs: Sequence[int], num_primes: int = 1000) -> list[int]:
    """
    Compute (a_{p_1}(E), ..., a_{p_num_primes}(E)).

    With num_primes=1000 this is the vector v_L(E) in Z^1000 from the paper
    (Eq. 3.6), using primes up to p_1000 = 7919.
    """
    E = _ellinit(ainvs)
    return [int(_pari.ellap(E, p)) for p in first_primes(num_primes)]
