"""
Prime helpers.

The paper enumerates primes p_1 = 2, p_2 = 3, ... and works with the first
1000 (p_1000 = 7919). We reuse a single PARI instance for speed.
"""

from functools import cache

import cypari2

_pari = cypari2.Pari()


@cache
def first_primes(n: int) -> tuple[int, ...]:
    """Return the first ``n`` primes as a tuple: (2, 3, 5, ...)."""
    # pari.primes(n) returns the first n primes as a PARI vector.
    return tuple(int(p) for p in _pari.primes(n))


def prime(i: int) -> int:
    """
    Return the i-th prime (1-indexed): prime(1) == 2, prime(1000) == 7919.
    """
    return int(_pari.prime(i))
