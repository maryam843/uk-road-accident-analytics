"""DfT road traffic counts (AADF by count point): loading, cleaning and collision matching.

Used by notebook 08 to turn collision counts into rates per vehicle-kilometre on major roads.

* AADF (annual average daily flow) is the average number of motor vehicles passing a count
  point per day over a year. Each major-road count point stands for a road *link* of known
  length, so ``vehicle-km = AADF x link_length_km x 365`` (DfT's own definition).
* Link lengths are published for major roads only (motorways and A roads), so exposure can
  only be computed there.
* Collisions are matched to the nearest count point on the **same road**, in the same year,
  within a distance cap, using British National Grid coordinates (metres).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# Road categories in the AADF file (metadata, "Data definitions").
ROAD_CATEGORIES: dict[str, str] = {
    "TM": "Trunk motorway",
    "PM": "Principal motorway",
    "TA": "Trunk A road",
    "PA": "Principal A road",
    "MB": "Minor road, class B",
    "MCU": "Minor road, class C or unclassified",
}
MAJOR_CATEGORIES: tuple[str, ...] = ("TM", "PM", "TA", "PA")

AADF_COLUMNS: list[str] = [
    "count_point_id", "year", "region_name", "local_authority_name", "road_name", "road_category",
    "road_type", "start_junction_road_name", "end_junction_road_name", "easting", "northing",
    "latitude", "longitude", "link_length_km", "estimation_method", "estimation_method_detailed",
    "all_motor_vehicles",
]

# STATS19 first_road_class codes (data guide): 1 Motorway, 2 A(M), 3 A.
STATS19_MAJOR_CLASSES: dict[int, str] = {1: "Motorway", 2: "A(M)", 3: "A"}

DAYS_PER_YEAR = 365

UNMATCHED_REASONS = ("missing road number", "no same-road count point", "beyond distance cap")


def normalise_road_name(names: pd.Series) -> pd.Series:
    """Canonical road names: upper case, no spaces, ``A57M`` -> ``A57(M)``, ``M6(T)`` -> ``M6``.

    The AADF file spells motorway-standard A roads both ``A57(M)`` and ``A57M``; the M6 toll
    road is ``M6(T)``, which STATS19 records as motorway number 6.
    """
    out = names.astype("string").str.upper().str.replace(r"\s+", "", regex=True)
    out = out.str.replace(r"^(A\d+)M$", r"\1(M)", regex=True)
    return out.str.replace(r"^(M\d+)\(T\)$", r"\1", regex=True)


def collision_road_name(road_class: pd.Series, road_number: pd.Series) -> pd.Series:
    """Road name in AADF format from STATS19 ``first_road_class`` code and ``first_road_number``.

    Class 1 (motorway) -> ``M<n>``, 2 (A(M)) -> ``A<n>(M)``, 3 (A) -> ``A<n>``. Other classes,
    or a missing / zero road number, give NA.
    """
    cls = pd.to_numeric(road_class, errors="coerce")
    num = pd.to_numeric(road_number, errors="coerce")
    ok = num.notna() & (num > 0)
    num_str = num.where(ok).astype("Int64").astype("string")
    name = pd.Series(pd.NA, index=road_class.index, dtype="string")
    name = name.mask(ok & (cls == 1), "M" + num_str)
    name = name.mask(ok & (cls == 2), "A" + num_str + "(M)")
    return name.mask(ok & (cls == 3), "A" + num_str)


def load_aadf(path: Path, years: tuple[int, int] | None = None) -> pd.DataFrame:
    """Read the AADF CSV (plain or zipped) with the columns this project uses.

    Args:
        path: ``dft_traffic_counts_aadf.zip`` (or the CSV inside it).
        years: optional inclusive ``(first, last)`` year filter.
    """
    df = pd.read_csv(path, usecols=lambda c: c in AADF_COLUMNS, low_memory=False)
    missing = sorted(set(AADF_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"AADF file is missing columns: {missing}")
    if years is not None:
        df = df[df["year"].between(*years)]
    return df.reset_index(drop=True)


def clean_aadf(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical road names, a major-road flag and yearly vehicle-km; drop unusable rows.

    Rows without coordinates, a positive link length (major roads only) or a non-negative
    AADF get ``vehicle_km = NaN`` and ``usable = False``; they are kept so they can be counted.
    """
    out = df.copy()
    out["road_name_norm"] = normalise_road_name(out["road_name"])
    out["is_major"] = out["road_category"].isin(MAJOR_CATEGORIES)
    out["road_group"] = np.select(
        [out["road_category"].isin(["TM", "PM"]), out["road_category"] == "TA",
         out["road_category"] == "PA"],
        ["Motorway", "Trunk A road", "Principal A road"], default="Minor road")
    length = pd.to_numeric(out["link_length_km"], errors="coerce")
    flow = pd.to_numeric(out["all_motor_vehicles"], errors="coerce")
    has_xy = out[["easting", "northing"]].notna().all(axis=1)
    out["usable"] = out["is_major"] & has_xy & (length > 0) & (flow >= 0)
    out["vehicle_km"] = vehicle_km(flow, length).where(out["usable"])
    return out


def vehicle_km(aadf: pd.Series | np.ndarray, link_length_km: pd.Series | np.ndarray,
               days: int = DAYS_PER_YEAR) -> pd.Series | np.ndarray:
    """Vehicle-kilometres per year on a link: AADF x link length (km) x days."""
    return aadf * link_length_km * days


def match_to_count_points(
    collisions: pd.DataFrame,
    points: pd.DataFrame,
    cap_m: float,
    year_col: str = "year",
) -> pd.DataFrame:
    """Nearest count point on the same road, in the same year, within ``cap_m`` metres.

    Args:
        collisions: columns ``road_name`` (NA if unknown), ``easting``, ``northing`` and
            ``year_col``; any index.
        points: one row per (count point, year) with ``count_point_id``, ``road_name``,
            ``easting``, ``northing`` and ``year_col``.
        cap_m: maximum distance in metres.

    Returns:
        Frame aligned with ``collisions`` with ``count_point_id`` (NA if unmatched),
        ``distance_m`` (distance to the nearest same-road point, even beyond the cap) and
        ``reason``: ``"matched"`` or one of :data:`UNMATCHED_REASONS`.
    """
    result = pd.DataFrame(index=collisions.index)
    result["count_point_id"] = pd.array([pd.NA] * len(collisions), dtype="Int64")
    result["distance_m"] = np.nan
    result["reason"] = np.where(collisions["road_name"].isna(), UNMATCHED_REASONS[0],
                                UNMATCHED_REASONS[1])
    candidates = {key: grp for key, grp in points.groupby([year_col, "road_name"], sort=False)}
    has_name = collisions["road_name"].notna()
    ids = np.full(len(collisions), -1, dtype="int64")
    dists = np.full(len(collisions), np.nan)
    pos_of = pd.Series(np.arange(len(collisions)), index=collisions.index)
    for key, grp in collisions[has_name].groupby([year_col, "road_name"], sort=False):
        cand = candidates.get(key)
        if cand is None:
            continue
        tree = cKDTree(cand[["easting", "northing"]].to_numpy(float))
        dist, nearest = tree.query(grp[["easting", "northing"]].to_numpy(float), k=1)
        rows = pos_of.loc[grp.index].to_numpy()
        dists[rows] = dist
        ids[rows] = cand["count_point_id"].to_numpy()[nearest]
    found = ~np.isnan(dists)
    result["distance_m"] = dists
    reason = result["reason"].to_numpy(dtype=object)
    reason[found] = np.where(dists[found] <= cap_m, "matched", UNMATCHED_REASONS[2])
    result["reason"] = reason
    matched = reason == "matched"
    result["count_point_id"] = pd.array(np.where(matched, ids, 0), dtype="Int64")
    result.loc[~matched, "count_point_id"] = pd.NA
    return result


def describe_road_names(names: pd.Series, n: int = 10) -> dict[str, int]:
    """Most common names that do not look like ``M<n>``, ``A<n>`` or ``A<n>(M)``."""
    odd = names[~names.astype("string").str.fullmatch(r"(M\d+|A\d+(\(M\))?)", na=False)]
    return {str(k): int(v) for k, v in odd.value_counts().head(n).items()}
