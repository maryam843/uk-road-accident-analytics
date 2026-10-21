"""Smoke tests: package and config import, and the cleaning pipeline on a tiny fixture."""

from pathlib import Path

import pandas as pd
import pytest

from src import clean, config

FIXTURES = Path(__file__).parent / "fixtures"


def test_config_paths_are_relative_to_repo() -> None:
    assert config.YEAR_MIN == 2005
    assert config.YEAR_MAX == 2019
    assert config.RANDOM_STATE == 42


def test_display_path_is_repo_relative(tmp_path: Path) -> None:
    figure = config.REPO_ROOT / "reports" / "figures" / "example.png"
    assert config.display_path(figure) == "reports/figures/example.png"
    assert config.display_path(tmp_path) == str(tmp_path)


def test_clean_run_on_tiny_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``clean.run`` end to end on 300 synthetic collisions; no downloads, nothing written to the repo."""
    raw = tmp_path / "raw"
    raw.mkdir()
    collisions = pd.read_csv(FIXTURES / "collisions_fixture.csv", dtype=str).head(300)
    collisions.to_csv(raw / config.COLLISION_RAW_FILE, index=False)
    casualties = pd.read_csv(FIXTURES / "casualties_fixture.csv", dtype=str)
    casualties.to_csv(raw / config.CASUALTY_RAW_FILE, index=False)
    pd.read_csv(FIXTURES / "data_guide_codes.csv").to_excel(raw / config.DATA_GUIDE_RAW_FILE, index=False)
    monkeypatch.setattr(config, "DASHBOARD_DATA_DIR", tmp_path / "dashboard")

    log = clean.run(data_dir=tmp_path, chunksize=100)

    processed = tmp_path / "processed"
    coll = pd.read_parquet(processed / config.COLLISIONS_PARQUET)
    cas = pd.read_parquet(processed / config.CASUALTIES_PARQUET)
    monthly = pd.read_parquet(processed / config.MONTHLY_PARQUET)
    assert 0 < len(coll) <= 300
    assert coll["collision_year"].between(config.YEAR_MIN, config.YEAR_MAX).all()
    assert not coll["collision_index"].duplicated().any()
    assert cas["collision_index"].isin(coll["collision_index"]).all()
    assert monthly["collisions"].sum() == len(coll)
    assert (processed / config.CLEAN_LOG_FILE).exists()
    assert len(log["dashboard_csv_bytes"]) == 7
    assert sorted(p.name for p in (tmp_path / "dashboard").iterdir()) == sorted(log["dashboard_csv_bytes"])
