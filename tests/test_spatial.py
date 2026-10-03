"""Tests for the H3 and Gi* helpers."""

from __future__ import annotations

import h3
import numpy as np
import pytest

from src import spatial


def test_cells_and_centroids_round_trip() -> None:
    lat, lon = np.array([51.5074, 53.4808]), np.array([-0.1278, -2.2426])
    cells = spatial.to_cells(lat, lon, 7)
    assert all(h3.get_resolution(c) == 7 for c in cells)
    cent = spatial.cell_centroids(list(cells))
    # A resolution-7 cell is a few km across, so its centre is close to the point.
    assert np.allclose(cent["lat"], lat, atol=0.03) and np.allclose(cent["lon"], lon, atol=0.03)
    ring = spatial.cell_boundary_lonlat(cells[0])
    assert ring[0] == ring[-1] and len(ring) == 7


def test_local_xy_km_scale() -> None:
    xy = spatial.local_xy_km(np.array([52.0, 53.0]), np.array([-1.0, -1.0]))
    assert xy[1, 1] - xy[0, 1] == pytest.approx(spatial.KM_PER_DEG_LAT)
    assert xy[0, 0] == pytest.approx(xy[1, 0])


def test_gi_star_finds_planted_cluster() -> None:
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.arange(20), np.arange(20))
    xy = np.c_[gx.ravel(), gy.ravel()].astype(float)
    values = rng.poisson(2, len(xy)).astype(float)
    planted = (gx.ravel() < 4) & (gy.ravel() < 4)
    values[planted] += 25
    w = spatial.knn_weights(xy, k=6)
    res = spatial.gi_star(values, w, permutations=199, seed=1)
    assert set(res["cluster"]) <= {"hot", "cold", "not significant"}
    hot = res["cluster"].to_numpy() == "hot"
    assert hot[planted].mean() > 0.5
    assert not hot[~planted & (gx.ravel() > 8) & (gy.ravel() > 8)].any()
    assert (res["q_bh"] >= res["p_two_sided"] - 1e-12).all()
    # Same seed, same result.
    again = spatial.gi_star(values, w, permutations=199, seed=1)
    assert np.array_equal(res["z_sim"].to_numpy(), again["z_sim"].to_numpy())
