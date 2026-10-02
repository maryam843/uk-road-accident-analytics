"""Generate synthetic STATS19-shaped fixtures for tests and offline smoke runs.

The column order follows the field list of the DfT Road Safety Open Dataset data guide
(2025 edition), and coded values are drawn from the real code list in
``data_guide_codes.csv``. Every row is synthetic: the numbers do not describe real
collisions and must never be reported as results.

Deliberate defects are planted so the cleaning tests have something to catch:
years outside 2005-2019, exact and conflicting duplicate collision indexes, ``-1`` and
"unknown (self reported)" codes, invalid speed limits, missing and zero coordinates,
orphan and duplicate casualty rows.

Usage::

    python tests/fixtures/make_fixture.py                      # ~2,000-row test fixture
    python tests/fixtures/make_fixture.py --n 150000 --out-dir /tmp/smoke/raw --official-names
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 42

COLLISION_HEADER = [
    "collision_index", "collision_year", "collision_ref_no", "location_easting_osgr",
    "location_northing_osgr", "longitude", "latitude", "police_force", "collision_severity",
    "enhanced_collision_severity", "number_of_vehicles", "number_of_casualties", "date",
    "day_of_week", "time", "local_authority_district", "local_authority_ons_district",
    "local_authority_highway", "local_authority_highway_current", "first_road_class",
    "first_road_number", "road_type", "speed_limit", "junction_detail_historic",
    "junction_detail", "junction_control", "second_road_class", "second_road_number",
    "pedestrian_crossing_human_control_historic",
    "pedestrian_crossing_physical_facilities_historic", "pedestrian_crossing",
    "light_conditions", "weather_conditions", "road_surface_conditions",
    "special_conditions_at_site", "carriageway_hazards_historic", "carriageway_hazards",
    "urban_or_rural_area", "did_police_officer_attend_scene_of_collision",
    "did_police_officer_attend_scene_of_accident", "trunk_road_flag",
    "lsoa_of_collision_location", "lsoa_of_accident_location", "collision_injury_based",
    "collision_adjusted_serious", "collision_adjusted_slight",
]
CASUALTY_HEADER = [
    "collision_index", "collision_year", "collision_ref_no", "vehicle_reference",
    "casualty_reference", "casualty_class", "sex_of_casualty", "age_of_casualty",
    "age_band_of_casualty", "casualty_severity", "enhanced_casualty_severity",
    "pedestrian_location", "pedestrian_movement", "car_passenger", "bus_or_coach_passenger",
    "pedestrian_road_maintenance_worker", "casualty_type", "casualty_imd_decile",
    "lsoa_of_casualty", "casualty_injury_based", "casualty_adjusted_serious",
    "casualty_adjusted_slight", "casualty_distance_banding", "escooter_flag",
]

SPEEDS = np.array([20, 30, 40, 50, 60, 70])
SPEED_P = np.array([0.05, 0.6, 0.1, 0.05, 0.12, 0.08])
SCOTLAND_OLD = [91, 92, 93, 94, 95, 96, 97, 98]


def _codes(guide: pd.DataFrame, table: str, field: str) -> np.ndarray:
    rows = guide[(guide["table"] == table) & (guide["field name"] == field)]
    codes = pd.to_numeric(rows["code/format"], errors="coerce").dropna().astype(int)
    return codes[codes >= 0].unique()


def _pick(rng: np.random.Generator, values: np.ndarray, n: int, p_missing: float = 0.01) -> np.ndarray:
    out = rng.choice(values, size=n)
    out[rng.random(n) < p_missing] = -1
    return out


def make_collisions(n: int, guide: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic collision table with ``COLLISION_HEADER`` columns."""
    years = np.arange(2004, 2021)
    year_w = np.linspace(1.25, 0.8, len(years))
    year = rng.choice(years, size=n, p=year_w / year_w.sum())
    month_w = np.array([0.95, 0.88, 0.95, 0.92, 1.0, 1.0, 1.03, 0.98, 1.02, 1.08, 1.1, 1.05])
    month = rng.choice(np.arange(1, 13), size=n, p=month_w / month_w.sum())
    day = np.minimum(rng.integers(1, 32, size=n), pd.to_datetime(
        {"year": year, "month": month, "day": 1}).dt.days_in_month.to_numpy())
    date = pd.to_datetime({"year": year, "month": month, "day": day})
    hour_w = np.array([2, 1, 1, 1, 1, 2, 4, 8, 10, 6, 6, 7, 7, 7, 7, 9, 11, 11, 8, 6, 4, 3, 3, 2])
    hour = rng.choice(np.arange(24), size=n, p=hour_w / hour_w.sum())
    minute = rng.integers(0, 60, size=n)
    time = pd.Series([f"{h:02d}:{m:02d}" for h, m in zip(hour, minute, strict=True)])
    time[rng.random(n) < 0.005] = ""

    england_wales = [c for c in _codes(guide, "collision", "police_force") if c < 90]
    force = rng.choice(england_wales + SCOTLAND_OLD, size=n)
    scot = np.isin(force, SCOTLAND_OLD)
    force[scot & (year >= 2019)] = 99

    speed = rng.choice(SPEEDS, size=n, p=SPEED_P)
    urban = np.where(speed <= 30, 1, rng.choice([1, 2], size=n, p=[0.3, 0.7]))
    dark = (hour < 7) | (hour >= 19)
    light = np.where(dark, rng.choice([4, 5, 6, 7], size=n, p=[0.6, 0.05, 0.3, 0.05]), 1)
    weather = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], size=n,
                         p=[0.80, 0.12, 0.01, 0.01, 0.01, 0.002, 0.008, 0.02, 0.02])
    surface = np.where(np.isin(weather, [2, 5]), 2, rng.choice([1, 2, 4], size=n, p=[0.75, 0.2, 0.05]))

    logit = (-2.4 + 0.012 * (speed - 30) + 0.5 * (urban == 2) + 0.35 * np.isin(light, [5, 6])
             + 0.2 * (hour < 5) + 0.03 * (year - 2005) * (year >= 2016))
    p_ksi = 1 / (1 + np.exp(-logit))
    ksi = rng.random(n) < p_ksi
    fatal = ksi & (rng.random(n) < 0.08 + 0.1 * (speed >= 60))
    severity = np.where(fatal, 1, np.where(ksi, 2, 3))

    injury_based = ((year >= 2016) & (rng.random(n) < 0.6)).astype(int)
    adj_serious = np.where(
        severity == 1, 0.0,
        np.where(injury_based == 1, (severity == 2).astype(float),
                 np.where(severity == 2, rng.uniform(0.75, 1.0, n), rng.uniform(0.0, 0.15, n))),
    )
    adj_slight = np.where(severity == 1, 0.0, 1 - adj_serious)

    lat = rng.uniform(50.2, 58.5, size=n).round(6)
    lon = rng.uniform(-5.5, 1.6, size=n).round(6)
    lat_s, lon_s = lat.astype(object), lon.astype(object)
    no_coord = rng.random(n) < 0.01
    lat_s[no_coord], lon_s[no_coord] = "", ""
    zero = rng.random(n) < 0.003
    lat_s[zero], lon_s[zero] = 0.0, 0.0

    speed_raw = speed.copy()
    bad = rng.random(n)
    speed_raw[bad < 0.004] = -1
    speed_raw[(bad >= 0.004) & (bad < 0.006)] = 99
    speed_raw[(bad >= 0.006) & (bad < 0.008)] = 15

    seq = np.arange(n)
    index = [f"{y}{f:02d}{s:07d}" for y, f, s in zip(year, force, seq, strict=True)]
    n_veh = rng.choice([1, 2, 3, 4], size=n, p=[0.3, 0.55, 0.11, 0.04])
    n_cas = rng.choice([1, 2, 3], size=n, p=[0.78, 0.17, 0.05])
    attend = _pick(rng, np.array([1, 2, 3]), n)

    df = pd.DataFrame({
        "collision_index": index,
        "collision_year": year,
        "collision_ref_no": [f"{f:02d}{s:07d}" for f, s in zip(force, seq, strict=True)],
        "location_easting_osgr": rng.integers(100000, 650000, size=n),
        "location_northing_osgr": rng.integers(10000, 1000000, size=n),
        "longitude": lon_s,
        "latitude": lat_s,
        "police_force": force,
        "collision_severity": severity,
        "enhanced_collision_severity": -1,
        "number_of_vehicles": n_veh,
        "number_of_casualties": n_cas,
        "date": date.dt.strftime("%d/%m/%Y"),
        "day_of_week": ((date.dt.dayofweek + 1) % 7 + 1).to_numpy(),
        "time": time,
        "local_authority_district": rng.integers(1, 600, size=n),
        "local_authority_ons_district": "E08000001",
        "local_authority_highway": "E08000001",
        "local_authority_highway_current": "E08000001",
        "first_road_class": _pick(rng, _codes(guide, "collision", "first_road_class"), n),
        "first_road_number": rng.integers(0, 6000, size=n),
        "road_type": _pick(rng, _codes(guide, "collision", "road_type"), n),
        "speed_limit": speed_raw,
        "junction_detail_historic": _pick(rng, _codes(guide, "collision", "junction_detail_historic"), n),
        "junction_detail": -1,
        "junction_control": _pick(rng, _codes(guide, "collision", "junction_control"), n),
        "second_road_class": _pick(rng, _codes(guide, "collision", "second_road_class"), n),
        "second_road_number": 0,
        "pedestrian_crossing_human_control_historic": 0,
        "pedestrian_crossing_physical_facilities_historic": 0,
        "pedestrian_crossing": -1,
        "light_conditions": light,
        "weather_conditions": weather,
        "road_surface_conditions": surface,
        "special_conditions_at_site": _pick(rng, _codes(guide, "collision", "special_conditions_at_site"), n),
        "carriageway_hazards_historic": 0,
        "carriageway_hazards": -1,
        "urban_or_rural_area": urban,
        "did_police_officer_attend_scene_of_collision": attend.astype(str).astype(object),
        "did_police_officer_attend_scene_of_accident": "",
        "trunk_road_flag": _pick(rng, np.array([1, 2]), n, 0.05),
        "lsoa_of_collision_location": "E01000001",
        "lsoa_of_accident_location": "",
        "collision_injury_based": injury_based,
        "collision_adjusted_serious": adj_serious.round(6),
        "collision_adjusted_slight": adj_slight.round(6),
    })
    # Police attendance: blank the collision-named field for a few rows and put the value
    # in the legacy accident-named field, so the coalescing path is exercised.
    legacy = rng.random(n) < 0.02
    df.loc[legacy, "did_police_officer_attend_scene_of_accident"] = df.loc[
        legacy, "did_police_officer_attend_scene_of_collision"].astype(str)
    df.loc[legacy, "did_police_officer_attend_scene_of_collision"] = ""

    # Duplicates in 2012: exact copies plus a few rows that reuse an index with other content.
    in_2012 = df.index[df["collision_year"] == 2012]
    n_exact = max(1, len(in_2012) // 10)
    exact = df.loc[rng.choice(in_2012, size=n_exact, replace=False)]
    conflict = df.loc[rng.choice(in_2012, size=max(1, n_exact // 5), replace=False)].copy()
    conflict["collision_severity"] = 3
    conflict["time"] = "12:00"
    return pd.concat([df, exact, conflict], ignore_index=True)[COLLISION_HEADER]


def make_casualties(collisions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic casualty table consistent with the collision severities."""
    base = collisions.drop_duplicates("collision_index")
    reps = base["number_of_casualties"].to_numpy()
    rows = base.loc[base.index.repeat(reps)].reset_index(drop=True)
    cas_ref = rows.groupby("collision_index").cumcount().to_numpy() + 1
    sev = rows["collision_severity"].to_numpy().copy()
    later = cas_ref > 1
    sev[later] = np.maximum(sev[later], rng.choice([2, 3], size=later.sum(), p=[0.2, 0.8]))
    n = len(rows)
    injury = rows["collision_injury_based"].to_numpy()
    adj_serious = np.where(sev == 1, 0.0, np.where(
        injury == 1, (sev == 2).astype(float),
        np.where(sev == 2, rng.uniform(0.75, 1.0, n), rng.uniform(0.0, 0.15, n))))
    df = pd.DataFrame({
        "collision_index": rows["collision_index"],
        "collision_year": rows["collision_year"],
        "collision_ref_no": rows["collision_ref_no"],
        "vehicle_reference": np.minimum(cas_ref, rows["number_of_vehicles"].to_numpy()),
        "casualty_reference": cas_ref,
        "casualty_class": rng.choice([1, 2, 3], size=n, p=[0.6, 0.27, 0.13]),
        "sex_of_casualty": _pick(rng, np.array([1, 2, 9]), n),
        "age_of_casualty": rng.integers(1, 90, size=n),
        "age_band_of_casualty": _pick(rng, np.arange(1, 12), n),
        "casualty_severity": sev,
        "enhanced_casualty_severity": -1,
        "pedestrian_location": 0,
        "pedestrian_movement": 0,
        "car_passenger": 0,
        "bus_or_coach_passenger": 0,
        "pedestrian_road_maintenance_worker": 0,
        "casualty_type": rng.choice([0, 1, 3, 5, 9, 11, 19], size=n),
        "casualty_imd_decile": rng.integers(1, 11, size=n),
        "lsoa_of_casualty": "E01000001",
        "casualty_injury_based": injury,
        "casualty_adjusted_serious": adj_serious.round(6),
        "casualty_adjusted_slight": np.where(sev == 1, 0.0, 1 - adj_serious).round(6),
        "casualty_distance_banding": -1,
        "escooter_flag": 0,
    })
    orphans = df.sample(n=max(1, n // 200), random_state=SEED).copy()
    orphans["collision_index"] = orphans["collision_index"].str.slice(0, 6) + "ORPHAN0"
    dup = df.sample(n=max(1, n // 300), random_state=SEED + 1)
    return pd.concat([df, orphans, dup], ignore_index=True)[CASUALTY_HEADER]


def write_ods(path: Path, sheets: dict[str, list[list[object]]]) -> None:
    """Write a minimal OpenDocument spreadsheet (enough for :func:`src.clean.read_ods`)."""
    body = []
    for name, rows in sheets.items():
        body.append(f'<table:table table:name="{escape(name)}">')
        for row in rows:
            body.append("<table:table-row>")
            for value in row:
                if isinstance(value, int | float) and not isinstance(value, bool):
                    body.append(
                        f'<table:table-cell office:value-type="float" office:value="{value}">'
                        f"<text:p>{value}</text:p></table:table-cell>"
                    )
                else:
                    body.append(
                        f'<table:table-cell office:value-type="string"><text:p>{escape(str(value))}'
                        "</text:p></table:table-cell>"
                    )
            body.append("</table:table-row>")
        body.append("</table:table>")
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<office:document-content '
        'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
        "<office:body><office:spreadsheet>" + "".join(body) + "</office:spreadsheet></office:body>"
        "</office:document-content>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.spreadsheet")
        zf.writestr("content.xml", content)


def ras0101_sheets(yearly: pd.DataFrame) -> dict[str, list[list[object]]]:
    """RAS0101-shaped sheets (same headers as the 2026 DfT release) from yearly totals."""
    coll = [["Reported road collisions, by severity, Great Britain, from 1926 [note 1]"],
            ["Year", "Fatal", "FSC (unadjusted)", "FSC (adjusted) [note 2]", "All collisions"],
            [1926, "[x]", "[x]", "[x]", 124000]]
    cas = [["Numbers and rates of reported road casualties by road user type"],
           ["Year", "Pedestrians (killed)", "All road users (killed)",
            "All road users (all severities) [note 6]", "", "Year", "Population (millions)"]]
    for year, r in yearly.iterrows():
        coll.append([int(year), int(r["fatal_collisions"]), int(r["ksi_collisions"]),
                     float(r["ksi_collisions_adjusted"]), int(r["collisions"])])
        cas.append([int(year), 1, int(r["killed"]), int(r["casualties"]), "", int(year), 60.0])
    return {"Cover_sheet": [["RAS0101"]], "Collisions": coll, "Casualties": cas}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1800, help="collisions before duplicates")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument(
        "--official-names", action="store_true",
        help="use the DfT raw file names (for a smoke run of python -m src.clean)",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    guide = pd.read_csv(HERE / "data_guide_codes.csv")
    collisions = make_collisions(args.n, guide, rng)
    casualties = make_casualties(collisions, rng)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.official_names:
        names = ("dft-road-casualty-statistics-collision-1979-latest-published-year.csv",
                 "dft-road-casualty-statistics-casualty-1979-latest-published-year.csv")
    else:
        names = ("collisions_fixture.csv", "casualties_fixture.csv")
    collisions.to_csv(args.out_dir / names[0], index=False)
    casualties.to_csv(args.out_dir / names[1], index=False)
    print(f"wrote {len(collisions)} collisions, {len(casualties)} casualties to {args.out_dir}")


if __name__ == "__main__":
    main()
