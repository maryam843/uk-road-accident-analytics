"""Project-wide configuration: paths, source URLs, study window and split definitions.

All paths are derived from the ``DATA_DIR`` environment variable, which defaults to
``<repo root>/data``. Nothing in this module points at an absolute location on disk.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    """Return the data directory (``DATA_DIR`` env var, default ``<repo>/data``)."""
    return Path(os.environ.get("DATA_DIR", REPO_ROOT / "data")).expanduser()


DATA_DIR: Path = get_data_dir()
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
KAGGLE_DIR: Path = DATA_DIR / "kaggle"
FIGURES_DIR: Path = Path(os.environ.get("FIGURES_DIR", REPO_ROOT / "reports" / "figures"))
DASHBOARD_DATA_DIR: Path = Path(
    os.environ.get("DASHBOARD_DATA_DIR", REPO_ROOT / "dashboard" / "data")
)
# Small result files written by notebooks 00, 03 and 07 and read by src/dashboard_exports.py.
RESULTS_DIR: Path = REPO_ROOT / "reports" / "results"

RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Sources (Department for Transport, OGL v3.0)
# ---------------------------------------------------------------------------
LANDING_PAGE_URL = "https://www.gov.uk/government/statistical-data-sets/road-safety-open-data"
RAS_TABLES_PAGE_URL = (
    "https://www.gov.uk/government/statistical-data-sets/"
    "reported-road-accidents-vehicles-and-casualties-tables-for-great-britain"
)
COLLISION_URL = (
    "https://data.dft.gov.uk/road-accidents-safety-data/"
    "dft-road-casualty-statistics-collision-1979-latest-published-year.csv"
)
CASUALTY_URL = (
    "https://data.dft.gov.uk/road-accidents-safety-data/"
    "dft-road-casualty-statistics-casualty-1979-latest-published-year.csv"
)
# Fallbacks used only if the landing pages cannot be parsed. Both were the links
# published on the gov.uk pages when this code was written; download.py tries to
# discover the current links first.
DATA_GUIDE_URL_FALLBACK = (
    "https://assets.publishing.service.gov.uk/media/6ab2a71d997a4b2950cced58/"
    "dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx"
)
RAS0101_URL_FALLBACK = (
    "https://assets.publishing.service.gov.uk/media/6a6398df9a419980593b2bdf/ras0101.ods"
)

# Road traffic statistics (DfT, OGL v3.0): annual average daily flow (AADF) by count point.
# download.py reads the current links from the downloads page; these are the links it showed
# when this code was written.
ROAD_TRAFFIC_DOWNLOADS_URL = "https://roadtraffic.dft.gov.uk/downloads"
AADF_URL_FALLBACK = (
    "https://storage.googleapis.com/dft-statistics/road-traffic/downloads/data-gov-uk/"
    "dft_traffic_counts_aadf.zip"
)
AADF_METADATA_URL_FALLBACK = (
    "https://storage.googleapis.com/dft-statistics/road-traffic/all-traffic-data-metadata.pdf"
)
AADF_RAW_FILE = "dft_traffic_counts_aadf.zip"
AADF_METADATA_RAW_FILE = "all-traffic-data-metadata.pdf"

COLLISION_RAW_FILE = "dft-road-casualty-statistics-collision-1979-latest-published-year.csv"
CASUALTY_RAW_FILE = "dft-road-casualty-statistics-casualty-1979-latest-published-year.csv"
DATA_GUIDE_RAW_FILE = "data_guide.xlsx"
RAS0101_RAW_FILE = "ras0101.ods"
DOWNLOAD_LOG_FILE = "DOWNLOAD_LOG.json"

# ---------------------------------------------------------------------------
# Study window and splits (locked decisions)
# ---------------------------------------------------------------------------
YEAR_MIN: int = 2005
YEAR_MAX: int = 2019

FORECAST_TRAIN_END = "2016-12"
FORECAST_TEST_START = "2017-01"
FORECAST_TEST_END = "2019-12"
BACKTEST_FIRST_ORIGIN = "2012-12"
BACKTEST_HORIZON: int = 12

CLF_TRAIN_YEARS: tuple[int, int] = (2005, 2015)
CLF_VALID_YEARS: tuple[int, int] = (2016, 2016)
CLF_TEST_YEARS: tuple[int, int] = (2017, 2019)

# ---------------------------------------------------------------------------
# Processed outputs
# ---------------------------------------------------------------------------
COLLISIONS_PARQUET = "collisions_2005_2019.parquet"
CASUALTIES_PARQUET = "casualties_2005_2019.parquet"
MONTHLY_PARQUET = "monthly_collisions_2005_2019.parquet"
YEARLY_PARQUET = "yearly_aggregates_2005_2019.parquet"
CLEAN_LOG_FILE = "CLEAN_LOG.json"

DASHBOARD_MAX_BYTES: int = 5 * 1024 * 1024

CHUNKSIZE: int = 500_000

VALID_SPEED_LIMITS: tuple[int, ...] = (20, 30, 40, 50, 60, 70)

# Rough bounding box for Great Britain, used only to flag (never impute) coordinates.
GB_LAT_RANGE: tuple[float, float] = (49.8, 61.0)
GB_LON_RANGE: tuple[float, float] = (-8.7, 1.9)

SEVERITY_ADJUSTED_FOOTNOTE = "adjusted for changes in police severity reporting (DfT)"
ATTRIBUTION = (
    "Contains public sector information licensed under the Open Government Licence v3.0. "
    "Source: Department for Transport, Road Safety Data (STATS19)."
)


def display_path(path: Path | str) -> str:
    """``path`` relative to the repo root when it is inside the repo, else unchanged (for logs)."""
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with a compact format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
