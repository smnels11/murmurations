"""
Access to the LMFDB read-only Postgres mirror.

The connection settings below are the PUBLIC, read-only credentials published
by the LMFDB collaboration (https://www.lmfdb.org/api/options). They are
documented constants, not secrets, so they live directly in code.

Elliptic-curve data over Q lives in ``ec_curvedata``. The columns we use:
    lmfdb_label   text        e.g. '11.a1'
    lmfdb_iso     text        isogeny class, e.g. '11.a'
    lmfdb_number  smallint    curve # within the class (==1 is a representative)
    conductor     integer
    rank          smallint    (isogeny-invariant)
    ainvs         numeric[]   minimal Weierstrass model [a1,a2,a3,a4,a6]

Because a_p and rank are isogeny-class invariants and the paper uses one curve
per isogeny class, we default to lmfdb_number = 1 (one representative per
class).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import psycopg

# Public read-only mirror credentials: https://www.lmfdb.org/api/options
_CONNINFO = (
    "host=devmirror.lmfdb.xyz port=5432 dbname=lmfdb user=lmfdb password=lmfdb"
)


def connect() -> psycopg.Connection:
    """
    Open a connection to the LMFDB mirror. Use as a context manager.
    """
    return psycopg.connect(_CONNINFO, connect_timeout=30)


def _to_int_list(ainvs) -> list[int]:
    return [int(a) for a in ainvs]


def count_curves(
    conductor_min: int,
    conductor_max: int,
    rank: int | None = None,
    one_per_class: bool = True,
) -> int:
    """
    Count curves matching the filters (sanity-check against the paper's counts).
    """
    where = ["conductor >= %s", "conductor <= %s"]
    params: list = [conductor_min, conductor_max]
    if rank is not None:
        where.append("rank = %s")
        params.append(rank)
    if one_per_class:
        where.append("lmfdb_number = 1")
    sql = f"SELECT count(*) FROM ec_curvedata WHERE {' AND '.join(where)}"
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return int(cur.fetchone()[0])


def fetch_curves(
    conductor_min: int,
    conductor_max: int,
    ranks: Sequence[int] | None = None,
    one_per_class: bool = True,
    limit: int | None = None,
    random_sample: bool = False,
    seed: float | None = None,
) -> pd.DataFrame:
    """
    Fetch curve metadata (label, iso, conductor, rank, ainvs) from the mirror.

    Parameters
    ----------
    ranks : restrict to these ranks (e.g. [0, 1, 2]); None means all.
    one_per_class : keep a single representative per isogeny class
                    (lmfdb_number=1).
    limit : cap the number of rows returned.
    random_sample : if True, order randomly before applying ``limit`` (useful
                    for the balanced random samples the paper draws). Uses
                    Postgres setseed.
    """
    where = ["conductor >= %s", "conductor <= %s"]
    params: list = [conductor_min, conductor_max]
    if ranks is not None:
        where.append("rank = ANY(%s)")
        params.append(list(ranks))
    if one_per_class:
        where.append("lmfdb_number = 1")

    order = (
        "ORDER BY random()"
        if random_sample
        else "ORDER BY conductor, iso_nlabel, lmfdb_number"
    )
    sql = (
        "SELECT lmfdb_label, lmfdb_iso, conductor, rank, "
        "torsion, class_size, ainvs "
        f"FROM ec_curvedata WHERE {' AND '.join(where)} {order}"
    )
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)

    with connect() as conn, conn.cursor() as cur:
        if random_sample and seed is not None:
            cur.execute("SELECT setseed(%s)", [seed])
        cur.execute(sql, params)
        rows = cur.fetchall()

    df = pd.DataFrame(
        rows,
        columns=[
            "lmfdb_label",
            "lmfdb_iso",
            "conductor",
            "rank",
            "torsion",
            "class_size",
            "ainvs",
        ],
    )
    df["conductor"] = df["conductor"].astype(int)
    df["rank"] = df["rank"].astype(int)
    # torsion = ORDER of E(Q)_tors (1 == trivial). NOTE: unlike rank and a_p,
    # torsion is NOT an isogeny-class invariant, so with one_per_class=True this
    # is the torsion of an arbitrary representative. class_size lets you find
    # (and exclude) the classes where that ambiguity bites.
    df["torsion"] = df["torsion"].astype(int)
    df["class_size"] = df["class_size"].astype(int)
    df["ainvs"] = df["ainvs"].apply(_to_int_list)
    return df
