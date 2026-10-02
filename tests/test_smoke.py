"""Smoke test: package and config import."""

from src import config


def test_config_paths_are_relative_to_repo() -> None:
    assert config.YEAR_MIN == 2005
    assert config.YEAR_MAX == 2019
    assert config.RANDOM_STATE == 42
