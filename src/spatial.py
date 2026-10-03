"""H3 aggregation and Getis-Ord Gi* hotspot helpers used by notebook 05.

A cell is called a hotspot only if its Gi* statistic is significantly positive after a
Benjamini-Hochberg false discovery rate correction across all cells tested.
"""

from __future__ import annotations

import warnings

import h3
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON_EQUATOR = 111.32


def to_cells(lat: np.ndarray | pd.Series, lon: np.ndarray | pd.Series, res: int) -> np.ndarray:
    """H3 cell id for every (lat, lon) pair at resolution ``res``."""
    lat_arr = np.asarray(lat, float)
    lon_arr = np.asarray(lon, float)
    return np.array([h3.latlng_to_cell(a, b, res) for a, b in zip(lat_arr, lon_arr, strict=True)])


def cell_centroids(cells: pd.Index | list[str]) -> pd.DataFrame:
    """Centre latitude and longitude of each H3 cell, indexed by cell id."""
    centres = [h3.cell_to_latlng(c) for c in cells]
    return pd.DataFrame(centres, index=pd.Index(cells, name="cell"), columns=["lat", "lon"])


def local_xy_km(lat: np.ndarray | pd.Series, lon: np.ndarray | pd.Series) -> np.ndarray:
    """Equirectangular projection to kilometres around the mean latitude.

    Accurate to well under 1% for neighbour searches at the scale of a few hexagons, which
    is all it is used for (k-nearest-neighbour weights).
    """
    lat_arr = np.asarray(lat, float)
    lon_arr = np.asarray(lon, float)
    scale = KM_PER_DEG_LON_EQUATOR * np.cos(np.deg2rad(lat_arr.mean()))
    return np.c_[lon_arr * scale, lat_arr * KM_PER_DEG_LAT]


def knn_weights(xy: np.ndarray, k: int):
    """k-nearest-neighbour spatial weights (libpysal), row-standardised later by Gi*.

    Warnings about disconnected components (islands far from the mainland) are silenced;
    every cell still has exactly ``k`` neighbours.
    """
    from libpysal.weights import KNN

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return KNN.from_array(np.asarray(xy, float), k=k)


def gi_star(
    values: np.ndarray | pd.Series,
    w,
    permutations: int = 999,
    seed: int = 42,
    alpha: float = 0.05,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Getis-Ord Gi* with permutation inference and Benjamini-Hochberg FDR control.

    The permutations (conditional randomisation, ``esda.G_Local(star=True)``) give the
    expected value and spread of Gi* under spatial randomness for each cell; ``z_sim`` is
    the observed statistic standardised by them. The two-sided p-value comes from
    ``z_sim`` (normal tail), because the pure permutation p-value cannot go below
    ``1 / (permutations + 1)``, which is too coarse for an FDR correction over tens of
    thousands of cells. ``p_sim`` (esda's one-sided pseudo p-value) is returned as well.

    Returns:
        Frame with ``value, gi, z_sim, p_sim, p_two_sided, q_bh, significant, cluster``
        where ``cluster`` is ``"hot"``, ``"cold"`` or ``"not significant"``.
    """
    from esda import G_Local

    y = np.asarray(values, float)
    with warnings.catch_warnings():
        # esda notes that it infers the self-weight (star) from the row maximum: intended.
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        g = G_Local(y, w, transform="R", star=True, permutations=permutations,
                    keep_simulations=True, n_jobs=1, seed=seed)
    z = np.asarray(g.z_sim, float)
    g.rGs = g.sim = None  # free the (cells x permutations) simulation matrix
    p_two = 2 * stats.norm.sf(np.abs(z))
    reject, q, _, _ = multipletests(p_two, alpha=alpha, method="fdr_bh")
    cluster = np.where(reject & (z > 0), "hot", np.where(reject & (z < 0), "cold", "not significant"))
    return pd.DataFrame(
        {
            "value": y,
            "gi": np.asarray(g.Gs, float),
            "z_sim": z,
            "p_sim": np.asarray(g.p_sim, float),
            "p_two_sided": p_two,
            "q_bh": q,
            "significant": reject,
            "cluster": cluster,
        },
        index=index,
    )


def cell_boundary_lonlat(cell: str, decimals: int = 5) -> list[list[float]]:
    """Closed GeoJSON ring ``[[lon, lat], ...]`` for an H3 cell."""
    ring = [[round(lon, decimals), round(lat, decimals)] for lat, lon in h3.cell_to_boundary(cell)]
    return ring + [ring[0]]
