"""Clean the DfT STATS19 collision and casualty tables for 2005-2019.

Usage::

    python -m src.clean

Steps (all logged, counts written to ``DATA_DIR/processed/CLEAN_LOG.json``):

1. Read the header first and normalise column names to snake_case ``collision_*``
   (older releases used ``accident_*``). Only the columns listed in
   :data:`COLLISION_COLUMNS` / :data:`CASUALTY_COLUMNS` are loaded.
2. Read the CSV in chunks (``config.CHUNKSIZE`` rows), keep collision years 2005-2019.
3. Drop duplicated collision indexes (keep first) and log how many there were.
4. Turn ``-1`` and "data missing or out of range" codes into NaN and add ``*_label``
   columns decoded with the DfT data guide.
5. Keep only valid speed limits (20/30/40/50/60/70); others become NaN.
6. Parse date + time, flag (never impute) invalid coordinates.
7. Keep only casualties whose collision survived the steps above.
8. Write Parquet outputs plus monthly and yearly aggregates.

This module also reads DfT table RAS0101 (``.ods``) with the standard library, so no
extra spreadsheet dependency is needed.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions (canonical, post-normalisation names)
# ---------------------------------------------------------------------------
COLLISION_CODED_FIELDS: list[str] = [
    "police_force",
    "collision_severity",
    "day_of_week",
    "first_road_class",
    "road_type",
    "junction_detail_historic",
    "junction_control",
    "second_road_class",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "special_conditions_at_site",
    "urban_or_rural_area",
    "did_police_officer_attend_scene_of_collision",
    "trunk_road_flag",
    "collision_injury_based",
]
COLLISION_NUMERIC_FIELDS: list[str] = [
    "collision_year",
    "speed_limit",
    "number_of_vehicles",
    "number_of_casualties",
    "longitude",
    "latitude",
    "collision_adjusted_serious",
    "collision_adjusted_slight",
]
COLLISION_TEXT_FIELDS: list[str] = ["collision_index", "date", "time"]
COLLISION_COLUMNS: list[str] = (
    COLLISION_TEXT_FIELDS + COLLISION_NUMERIC_FIELDS + COLLISION_CODED_FIELDS
)

CASUALTY_CODED_FIELDS: list[str] = [
    "casualty_class",
    "casualty_severity",
    "casualty_type",
    "sex_of_casualty",
    "age_band_of_casualty",
    "casualty_injury_based",
]
CASUALTY_NUMERIC_FIELDS: list[str] = [
    "collision_year",
    "vehicle_reference",
    "casualty_reference",
    "casualty_adjusted_serious",
    "casualty_adjusted_slight",
]
CASUALTY_TEXT_FIELDS: list[str] = ["collision_index"]
CASUALTY_COLUMNS: list[str] = CASUALTY_TEXT_FIELDS + CASUALTY_NUMERIC_FIELDS + CASUALTY_CODED_FIELDS
CASUALTY_KEY: list[str] = ["collision_index", "vehicle_reference", "casualty_reference"]

# Fields present in the file but deliberately not used, with the reason. Rendered in
# notebooks so the exclusions are documented next to the analysis.
EXCLUDED_FIELDS: dict[str, str] = {
    "junction_detail": "2024 STATS19 specification; for 2005-2019 roundabout, mini-roundabout "
    "and slip road are all converted to code 0, so junction_detail_historic is used instead",
    "pedestrian_crossing": "2024 specification; historic equivalents split into two fields",
    "carriageway_hazards": "2024 specification; not used in this project",
    "enhanced_collision_severity": "only collected from 2023",
    "local_authority_highway_current": "boundary-dependent; not needed for 2005-2019 analysis",
    "lsoa_of_collision_location": "small-area code, not used (privacy and size)",
}

# Codes whose label means "missing" become NaN. "-1" is always treated as missing.
MISSING_LABEL_PATTERN = re.compile(r"data missing or out of range|unknown \(self rep", re.I)
LEGACY_RENAMES: dict[str, str] = {
    "collision_reference": "collision_ref_no",
    "collision_ref": "collision_ref_no",
}

SEVERITY_FATAL, SEVERITY_SERIOUS, SEVERITY_SLIGHT = 1, 2, 3


# ---------------------------------------------------------------------------
# Column-name normalisation
# ---------------------------------------------------------------------------
def to_snake_case(name: str) -> str:
    """Lower-case ``name`` and replace any run of non-alphanumerics with ``_``."""
    name = name.replace("﻿", "").strip()
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)  # camelCase -> camel_Case
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def normalise_column_name(name: str) -> str:
    """Map a raw header name to the canonical snake_case ``collision_*`` form.

    Examples:
        ``Accident_Index`` -> ``collision_index``;
        ``did_police_officer_attend_scene_of_accident`` ->
        ``did_police_officer_attend_scene_of_collision``;
        ``accident_reference`` -> ``collision_ref_no``.
    """
    snake = to_snake_case(name)
    snake = re.sub(r"(^|_)accident(_|$)", r"\1collision\2", snake)
    return LEGACY_RENAMES.get(snake, snake)


def build_column_map(raw_columns: Iterable[str]) -> dict[str, str]:
    """Return ``{raw name: canonical name}``.

    If two raw columns normalise to the same canonical name (the file carries both an
    ``accident_*`` and a ``collision_*`` version), the one already spelt ``collision``
    keeps the canonical name and the other gets a ``__dup<n>`` suffix so it can be
    coalesced into the first by :func:`coalesce_duplicates`.
    """
    raw_columns = list(raw_columns)
    canonical = {raw: normalise_column_name(raw) for raw in raw_columns}
    # Columns already in canonical form win ties.
    ordered = sorted(raw_columns, key=lambda r: to_snake_case(r) != canonical[r])
    seen: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for raw in ordered:
        name = canonical[raw]
        if name in seen:
            seen[name] += 1
            mapping[raw] = f"{name}__dup{seen[name]}"
        else:
            seen[name] = 0
            mapping[raw] = name
    return {raw: mapping[raw] for raw in raw_columns}


def coalesce_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps in each canonical column from its ``__dup<n>`` twins, then drop them."""
    dup_cols = [c for c in df.columns if "__dup" in c]
    for dup in dup_cols:
        base = dup.split("__dup")[0]
        if base in df.columns:
            df[base] = df[base].fillna(df[dup])
        else:
            df[base] = df[dup]
    return df.drop(columns=dup_cols)


def resolve_usecols(header: Iterable[str], wanted: Iterable[str]) -> dict[str, str]:
    """Subset of :func:`build_column_map` for the canonical columns in ``wanted``."""
    wanted_set = set(wanted)
    col_map = build_column_map(header)
    selected = {
        raw: canon for raw, canon in col_map.items() if canon.split("__dup")[0] in wanted_set
    }
    found = {c.split("__dup")[0] for c in selected.values()}
    missing = sorted(wanted_set - found)
    if missing:
        logger.warning("Columns not found in header (will be absent): %s", missing)
    return selected


# ---------------------------------------------------------------------------
# Data guide code list
# ---------------------------------------------------------------------------
def load_code_list(path: Path) -> pd.DataFrame:
    """Load the DfT data guide code list as ``table, field, code, label``.

    Accepts the official ``.xlsx`` (the sheet whose columns are
    ``table / field name / code/format / label``) or a CSV with the same columns.
    Field names are normalised with :func:`normalise_column_name`; rows whose code is
    not an integer (formats such as ``(DD/MM/YYYY)``) are dropped. A blank label on
    code 0 (for example ``special_conditions_at_site``) is read as ``"None"``.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = None
        for sheet, df in pd.read_excel(path, sheet_name=None).items():
            cols = {str(c).strip().lower() for c in df.columns}
            if {"table", "field name", "code/format", "label"} <= cols:
                logger.info("Using code list sheet %r", sheet)
                frame = df
                break
        if frame is None:
            raise ValueError(f"No code-list sheet found in {path}")
    else:
        frame = pd.read_csv(path)
    frame = frame.rename(columns=lambda c: str(c).strip().lower())
    out = pd.DataFrame(
        {
            "table": frame["table"].astype(str).map(normalise_column_name),
            "field": frame["field name"].astype(str).map(normalise_column_name),
            "code": pd.to_numeric(frame["code/format"], errors="coerce"),
            "label": frame["label"],
        }
    )
    out = out.dropna(subset=["code"])
    out = out[out["code"] == out["code"].round()]
    out["code"] = out["code"].astype("int64")
    out["label"] = out["label"].where(out["label"].notna() | (out["code"] != 0), "None")
    out["label"] = out["label"].astype("string").str.strip()
    # A few codes appear twice with different labels (casualty_type 23); join them.
    out = (
        out.groupby(["table", "field", "code"], sort=False, dropna=False)["label"]
        .agg(lambda s: " / ".join(dict.fromkeys(s.dropna())) or pd.NA)
        .reset_index()
    )
    out["label"] = out["label"].astype("string")
    return out


def missing_codes_for(code_list: pd.DataFrame, table: str, field: str) -> set[int]:
    """Codes that mean "missing" for one field: -1 plus any code labelled as missing."""
    rows = code_list[(code_list["table"] == table) & (code_list["field"] == field)]
    flagged = rows.loc[rows["label"].fillna("").str.contains(MISSING_LABEL_PATTERN), "code"]
    return {-1, *map(int, flagged)}


def decode_field(
    values: pd.Series, code_list: pd.DataFrame, table: str, field: str
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """Set missing codes to NA and decode the rest into labels.

    Returns:
        ``(codes, labels, stats)`` where ``codes`` is nullable ``Int16``, ``labels`` is a
        ``category`` series and ``stats`` counts missing and unmapped values.
    """
    codes = pd.to_numeric(values, errors="coerce").astype("Int64")
    is_missing = codes.isin(list(missing_codes_for(code_list, table, field)))
    codes = codes.mask(is_missing)
    rows = code_list[(code_list["table"] == table) & (code_list["field"] == field)]
    lookup = dict(zip(rows["code"].astype(int), rows["label"], strict=True))
    labels = codes.map(lookup)
    unmapped = codes.notna() & labels.isna()
    stats = {"missing_to_nan": int(is_missing.sum()), "unmapped_codes": int(unmapped.sum())}
    if unmapped.any():
        logger.warning(
            "%s.%s: %d values with codes not in the data guide: %s",
            table,
            field,
            int(unmapped.sum()),
            sorted(codes[unmapped].unique().tolist())[:10],
        )
    return codes.astype("Int16"), labels.astype("category"), stats


def decode_codes(
    df: pd.DataFrame, code_list: pd.DataFrame, table: str, fields: Iterable[str]
) -> dict[str, dict[str, int]]:
    """Decode every field in ``fields`` present in ``df`` in place; return stats."""
    stats: dict[str, dict[str, int]] = {}
    for field in fields:
        if field not in df.columns:
            continue
        codes, labels, field_stats = decode_field(df[field], code_list, table, field)
        df[field] = codes
        df[f"{field}_label"] = labels
        stats[field] = field_stats
    return stats


# ---------------------------------------------------------------------------
# Row-level cleaning steps
# ---------------------------------------------------------------------------
def filter_years(
    df: pd.DataFrame, year_col: str = "collision_year", lo: int = config.YEAR_MIN,
    hi: int = config.YEAR_MAX,
) -> pd.DataFrame:
    """Keep rows whose ``year_col`` lies in ``[lo, hi]``."""
    years = pd.to_numeric(df[year_col], errors="coerce")
    return df.loc[years.between(lo, hi)].copy()


def dedupe_collisions(df: pd.DataFrame, key: str = "collision_index") -> tuple[pd.DataFrame, dict]:
    """Drop rows whose collision index was already seen (keep the first).

    Returns the de-duplicated frame and counts of exact duplicate rows and of rows that
    share an index but differ in content (those are logged as a warning).
    """
    dup_index = df.duplicated(subset=[key], keep="first")
    exact = df.duplicated(keep="first")
    stats = {
        "duplicate_index_rows_dropped": int(dup_index.sum()),
        "exact_duplicate_rows": int((dup_index & exact).sum()),
        "conflicting_duplicate_rows": int((dup_index & ~exact).sum()),
    }
    if stats["duplicate_index_rows_dropped"]:
        logger.info("Dropped %d duplicate %s rows", stats["duplicate_index_rows_dropped"], key)
    if stats["conflicting_duplicate_rows"]:
        logger.warning(
            "%d rows share a %s with an earlier row but differ in content",
            stats["conflicting_duplicate_rows"],
            key,
        )
    return df.loc[~dup_index].copy(), stats


def validate_speed_limit(values: pd.Series) -> tuple[pd.Series, int]:
    """Keep 20/30/40/50/60/70 mph; set everything else to NA.

    Returns the cleaned ``Int16`` series and the number of non-missing values that
    were invalid (missing codes such as -1 are counted too, since they are not valid).
    """
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.isin(config.VALID_SPEED_LIMITS)
    n_invalid = int((numeric.notna() & ~valid).sum())
    if n_invalid:
        logger.info("speed_limit: %d values outside %s set to NaN", n_invalid, config.VALID_SPEED_LIMITS)
    return numeric.where(valid).astype("Int16"), n_invalid


def parse_datetime(df: pd.DataFrame) -> dict[str, int]:
    """Add ``date`` (datetime64), ``hour`` and ``datetime`` columns in place.

    ``date`` is read as DD/MM/YYYY (the DfT format) with an ISO fallback; ``time`` as
    HH:MM. A missing time leaves ``hour`` and ``datetime`` empty but keeps ``date``.
    """
    raw_date = df["date"].astype("string")
    date = pd.to_datetime(raw_date, format="%d/%m/%Y", errors="coerce")
    iso = pd.to_datetime(raw_date.where(date.isna()), format="%Y-%m-%d", errors="coerce")
    date = date.fillna(iso)
    time_str = df["time"].astype("string").str.strip().str.slice(0, 5)
    parts = time_str.str.extract(r"^(\d{1,2}):(\d{2})$")
    hour = pd.to_numeric(parts[0], errors="coerce")
    minute = pd.to_numeric(parts[1], errors="coerce")
    ok_time = (hour.between(0, 23) & minute.between(0, 59)).fillna(False).astype(bool)
    hour = hour.where(ok_time)
    minute = minute.where(ok_time)
    df["date"] = date
    df["hour"] = hour.astype("Int8")
    df["datetime"] = date + pd.to_timedelta(hour * 60 + minute, unit="min")
    return {"unparsed_dates": int(date.isna().sum()), "missing_or_bad_times": int((~ok_time).sum())}


def flag_coordinates(df: pd.DataFrame) -> int:
    """Add boolean ``coord_valid`` (inside a GB bounding box, not missing, not 0).

    Invalid coordinates are flagged only; they are never imputed.
    """
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    valid = lat.between(*config.GB_LAT_RANGE) & lon.between(*config.GB_LON_RANGE)
    df["coord_valid"] = valid.fillna(False).astype(bool)
    n_invalid = int((~df["coord_valid"]).sum())
    logger.info("Flagged %d collisions with missing or out-of-range coordinates", n_invalid)
    return n_invalid


def weekday_mismatches(df: pd.DataFrame) -> int:
    """Count rows whose ``day_of_week`` code (1 = Sunday) disagrees with ``date``."""
    from_date = (df["date"].dt.dayofweek + 1) % 7 + 1
    differs = (df["day_of_week"].astype("Float64") != from_date).fillna(False).astype(bool)
    return int((df["day_of_week"].notna() & from_date.notna() & differs).sum())


def _read_filtered_chunks(
    path: Path, wanted: list[str], numeric: list[str], chunksize: int, keep_index: pd.Index | None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Chunked read: rename, coalesce, coerce numerics, filter years (and collision set)."""
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    col_map = resolve_usecols(header, wanted)
    stats = {"rows_read": 0, "non_numeric_values_coerced": 0}
    kept: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        usecols=list(col_map),
        dtype=str,
        chunksize=chunksize,
        encoding="utf-8-sig",
        encoding_errors="replace",
        na_values=["", "NULL", "NA", "NaN"],
        keep_default_na=False,
    )
    for i, chunk in enumerate(reader):
        stats["rows_read"] += len(chunk)
        chunk = coalesce_duplicates(chunk.rename(columns=col_map))
        for col in numeric:
            if col in chunk.columns:
                converted = pd.to_numeric(chunk[col], errors="coerce")
                stats["non_numeric_values_coerced"] += int(
                    (chunk[col].notna() & converted.isna()).sum()
                )
                chunk[col] = converted
        if "collision_year" not in chunk.columns:
            chunk["collision_year"] = pd.to_datetime(
                chunk["date"], format="%d/%m/%Y", errors="coerce"
            ).dt.year
        chunk = filter_years(chunk)
        if keep_index is not None:
            chunk = chunk.loc[chunk["collision_index"].isin(keep_index)]
        kept.append(chunk)
        logger.info("chunk %d: %d rows kept (%d read so far)", i, len(chunk), stats["rows_read"])
    df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=wanted)
    stats["rows_in_study_window"] = len(df)
    return df, stats


def process_collisions(
    path: Path, code_list: pd.DataFrame, chunksize: int = config.CHUNKSIZE
) -> tuple[pd.DataFrame, dict]:
    """Run every collision cleaning step and return ``(collisions, stats)``."""
    coded = list(COLLISION_CODED_FIELDS)
    df, stats = _read_filtered_chunks(
        path, COLLISION_COLUMNS, COLLISION_NUMERIC_FIELDS + coded, chunksize, keep_index=None
    )
    df, stats["dedup"] = dedupe_collisions(df)
    stats["decode"] = decode_codes(df, code_list, "collision", coded)
    df["speed_limit"], stats["invalid_speed_limit"] = validate_speed_limit(df["speed_limit"])
    stats["datetime"] = parse_datetime(df)
    stats["weekday_code_vs_date_mismatch"] = weekday_mismatches(df)
    stats["invalid_coordinates_flagged"] = flag_coordinates(df)
    df["collision_year"] = df["collision_year"].astype("Int16")
    for col in ("number_of_vehicles", "number_of_casualties"):
        df[col] = df[col].astype("Int16")
    for col in ("collision_adjusted_serious", "collision_adjusted_slight"):
        if col in df.columns:
            df[col] = df[col].astype("float32")
    sev = df["collision_severity"]
    df["ksi"] = sev.isin([SEVERITY_FATAL, SEVERITY_SERIOUS]).astype("Int8").mask(sev.isna())
    df["collision_index"] = df["collision_index"].astype("string")
    stats["rows_out"] = len(df)
    return df.reset_index(drop=True), stats


def process_casualties(
    path: Path,
    code_list: pd.DataFrame,
    collision_index: pd.Index,
    chunksize: int = config.CHUNKSIZE,
) -> tuple[pd.DataFrame, dict]:
    """Clean casualties, keeping only those whose collision is in ``collision_index``."""
    df, stats = _read_filtered_chunks(
        path,
        CASUALTY_COLUMNS,
        CASUALTY_NUMERIC_FIELDS + CASUALTY_CODED_FIELDS,
        chunksize,
        keep_index=pd.Index(collision_index),
    )
    dup = df.duplicated(subset=CASUALTY_KEY, keep="first")
    stats["duplicate_casualty_rows_dropped"] = int(dup.sum())
    df = df.loc[~dup].copy()
    stats["decode"] = decode_codes(df, code_list, "casualty", CASUALTY_CODED_FIELDS)
    for col in ("collision_year", "vehicle_reference", "casualty_reference"):
        df[col] = df[col].astype("Int16")
    for col in ("casualty_adjusted_serious", "casualty_adjusted_slight"):
        if col in df.columns:
            df[col] = df[col].astype("float32")
    df["collision_index"] = df["collision_index"].astype("string")
    stats["rows_out"] = len(df)
    return df.reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------
def _is(values: pd.Series, code: int) -> pd.Series:
    """``values == code`` as a plain bool series (NA compares as False)."""
    return values.eq(code).fillna(False).astype(bool)


def _severity_counts(df: pd.DataFrame, by: str | pd.Series) -> pd.DataFrame:
    """Collision counts by severity plus the adjusted fatal-or-serious count."""
    sev = df["collision_severity"]
    non_fatal = ~_is(sev, SEVERITY_FATAL)
    adj = df.get("collision_adjusted_serious", pd.Series(np.nan, index=df.index))
    frame = pd.DataFrame(
        {
            "collisions": 1,
            "fatal_collisions": _is(sev, SEVERITY_FATAL).astype(int),
            "serious_collisions": _is(sev, SEVERITY_SERIOUS).astype(int),
            "slight_collisions": _is(sev, SEVERITY_SLIGHT).astype(int),
            "serious_collisions_adjusted": adj.where(non_fatal, 0.0).astype(float),
        },
        index=df.index,
    )
    out = frame.groupby(by, observed=True).sum()
    out["ksi_collisions"] = out["fatal_collisions"] + out["serious_collisions"]
    out["ksi_collisions_adjusted"] = out["fatal_collisions"] + out["serious_collisions_adjusted"]
    return out


def monthly_series(collisions: pd.DataFrame) -> pd.DataFrame:
    """Monthly collision counts 2005-01..2019-12 with an explicit month-start index.

    Months with no collisions (none expected in real data) are filled with 0 so the
    series has no gaps.
    """
    month = collisions["date"].dt.to_period("M").dt.to_timestamp()
    out = _severity_counts(collisions, month)
    full = pd.date_range(f"{config.YEAR_MIN}-01-01", f"{config.YEAR_MAX}-12-01", freq="MS")
    out = out.reindex(full, fill_value=0)
    out.index.name = "month"
    return out


def yearly_aggregates(collisions: pd.DataFrame, casualties: pd.DataFrame) -> pd.DataFrame:
    """Yearly collision and casualty totals, unadjusted and severity-adjusted."""
    coll = _severity_counts(collisions, collisions["collision_year"].astype(int))
    sev = casualties["casualty_severity"]
    non_fatal = ~_is(sev, SEVERITY_FATAL)
    adj_serious = casualties.get("casualty_adjusted_serious", pd.Series(np.nan, index=casualties.index))
    adj_slight = casualties.get("casualty_adjusted_slight", pd.Series(np.nan, index=casualties.index))
    cas = pd.DataFrame(
        {
            "casualties": 1,
            "killed": _is(sev, SEVERITY_FATAL).astype(int),
            "serious_casualties": _is(sev, SEVERITY_SERIOUS).astype(int),
            "slight_casualties": _is(sev, SEVERITY_SLIGHT).astype(int),
            "serious_casualties_adjusted": adj_serious.where(non_fatal, 0.0).astype(float),
            "slight_casualties_adjusted": adj_slight.where(non_fatal, 0.0).astype(float),
        }
    ).groupby(casualties["collision_year"].astype(int)).sum()
    cas["ksi_casualties"] = cas["killed"] + cas["serious_casualties"]
    cas["ksi_casualties_adjusted"] = cas["killed"] + cas["serious_casualties_adjusted"]
    out = coll.join(cas, how="outer").fillna(0)
    count_cols = [c for c in out.columns if not c.endswith("_adjusted")]
    out[count_cols] = out[count_cols].astype("int64")
    out.index.name = "year"
    return out


# ---------------------------------------------------------------------------
# RAS0101 (official yearly totals)
# ---------------------------------------------------------------------------
_ODS_NS = {
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}


def read_ods(path: Path) -> dict[str, list[list[str]]]:
    """Read every sheet of an ``.ods`` file into lists of string cells.

    Uses the numeric ``office:value`` attribute when present (full precision),
    otherwise the cell text. Long runs of repeated empty cells are truncated.
    """
    t = "{" + _ODS_NS["table"] + "}"
    o = "{" + _ODS_NS["office"] + "}"
    p = "{" + _ODS_NS["text"] + "}p"
    root = ElementTree.fromstring(zipfile.ZipFile(path).read("content.xml"))
    sheets: dict[str, list[list[str]]] = {}
    for table in root.iter(f"{t}table"):
        rows: list[list[str]] = []
        for row in table.iter(f"{t}table-row"):
            cells: list[str] = []
            for cell in row:
                if cell.tag not in (f"{t}table-cell", f"{t}covered-table-cell"):
                    continue
                repeat = int(cell.get(f"{t}number-columns-repeated", "1"))
                value = cell.get(f"{o}value")
                if value is None:
                    value = " ".join("".join(x.itertext()) for x in cell.findall(p))
                cells.extend([value] * (repeat if value else min(repeat, 64)))
            while cells and cells[-1] == "":
                cells.pop()
            if cells:
                rows.append(cells)
        sheets[table.get(f"{t}name", "")] = rows
    return sheets


def _sheet_table(rows: list[list[str]]) -> pd.DataFrame:
    """First table on a sheet: header row starts with 'Year', stops at a blank column."""
    start = next(i for i, r in enumerate(rows) if r and r[0].strip().lower() == "year")
    header = rows[start]
    width = header.index("") if "" in header else len(header)
    body = [(r + [""] * width)[:width] for r in rows[start + 1 :]]
    return pd.DataFrame(body, columns=header[:width])


RAS0101_COLUMN_PATTERNS: dict[str, tuple[str, str]] = {
    # output column: (sheet, regex on the header text)
    "collisions": ("Collisions", r"^all collisions"),
    "fatal_collisions": ("Collisions", r"^fatal"),
    "ksi_collisions": ("Collisions", r"^fsc \(unadjusted\)"),
    "ksi_collisions_adjusted": ("Collisions", r"^fsc \(adjusted\)"),
    "killed": ("Casualties", r"^all road users \(killed\)"),
    "casualties": ("Casualties", r"^all road users \(all severities\)"),
}


def load_ras0101(path: Path) -> pd.DataFrame:
    """Return RAS0101 yearly totals indexed by year.

    Columns: ``collisions``, ``fatal_collisions``, ``ksi_collisions`` (fatal or serious,
    "FSC" in DfT terms), ``ksi_collisions_adjusted``, ``killed``, ``casualties``. RAS0101
    does not publish serious *casualties* separately, so serious is reconciled at
    collision level (``ksi_collisions - fatal_collisions``).

    Raises:
        KeyError: if a sheet or header cannot be matched; the message lists what exists.
    """
    sheets = read_ods(Path(path))
    tables: dict[str, pd.DataFrame] = {}
    out: dict[str, pd.Series] = {}
    for column, (sheet, pattern) in RAS0101_COLUMN_PATTERNS.items():
        if sheet not in sheets:
            raise KeyError(f"Sheet {sheet!r} not in RAS0101; sheets: {list(sheets)}")
        if sheet not in tables:
            tables[sheet] = _sheet_table(sheets[sheet])
        table = tables[sheet]
        matches = [c for c in table.columns if re.search(pattern, c.strip(), flags=re.I)]
        if not matches:
            raise KeyError(f"No column matching {pattern!r} on {sheet!r}: {list(table.columns)}")
        years = pd.to_numeric(table.iloc[:, 0].str.extract(r"^(\d{4})")[0], errors="coerce")
        values = pd.to_numeric(table[matches[0]].str.replace(",", ""), errors="coerce")
        out[column] = pd.Series(values.to_numpy(), index=years.to_numpy()).loc[lambda s: s.index.notna()]
    result = pd.DataFrame(out)
    result.index = result.index.astype(int)
    result.index.name = "year"
    return result.sort_index()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(data_dir: Path | None = None, chunksize: int = config.CHUNKSIZE) -> dict:
    """Run the full cleaning pipeline from ``DATA_DIR/raw`` to ``DATA_DIR/processed``."""
    data_dir = Path(data_dir) if data_dir else config.get_data_dir()
    raw, processed = data_dir / "raw", data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    code_list = load_code_list(raw / config.DATA_GUIDE_RAW_FILE)

    collisions, coll_stats = process_collisions(raw / config.COLLISION_RAW_FILE, code_list, chunksize)
    collisions.to_parquet(processed / config.COLLISIONS_PARQUET, index=False)
    logger.info("Wrote %d collisions", len(collisions))

    casualties, cas_stats = process_casualties(
        raw / config.CASUALTY_RAW_FILE, code_list, collisions["collision_index"], chunksize
    )
    casualties.to_parquet(processed / config.CASUALTIES_PARQUET, index=False)
    logger.info("Wrote %d casualties", len(casualties))

    monthly_series(collisions).to_parquet(processed / config.MONTHLY_PARQUET)
    yearly_aggregates(collisions, casualties).to_parquet(processed / config.YEARLY_PARQUET)

    log = {"collisions": coll_stats, "casualties": cas_stats, "excluded_fields": EXCLUDED_FIELDS}
    (processed / config.CLEAN_LOG_FILE).write_text(json.dumps(log, indent=2, default=int))
    return log


def main() -> None:
    """CLI entry point: ``python -m src.clean``."""
    config.setup_logging()
    run()


if __name__ == "__main__":
    main()
