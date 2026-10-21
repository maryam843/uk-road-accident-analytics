"""Smoke test: package and config import."""

from pathlib import Path

from src import config


def test_config_paths_are_relative_to_repo() -> None:
    assert config.YEAR_MIN == 2005
    assert config.YEAR_MAX == 2019
    assert config.RANDOM_STATE == 42


def test_display_path_is_repo_relative(tmp_path: Path) -> None:
    figure = config.REPO_ROOT / "reports" / "figures" / "example.png"
    assert config.display_path(figure) == "reports/figures/example.png"
    assert config.display_path(tmp_path) == str(tmp_path)
