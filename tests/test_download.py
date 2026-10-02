"""Tests for the offline parts of src.download (no network access needed)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src import download


def test_expected_total_full_and_partial() -> None:
    full = SimpleNamespace(status_code=200, headers={"Content-Length": "100"})
    assert download._expected_total(full, already=0) == 100
    partial = SimpleNamespace(status_code=206, headers={"Content-Range": "bytes 40-99/100"})
    assert download._expected_total(partial, already=40) == 100
    no_range = SimpleNamespace(status_code=206, headers={"Content-Length": "60"})
    assert download._expected_total(no_range, already=40) == 100


def test_append_log_replaces_entry(tmp_path: Path) -> None:
    download.append_log({"file": "a.csv", "bytes": 1}, tmp_path)
    download.append_log({"file": "b.csv", "bytes": 2}, tmp_path)
    download.append_log({"file": "a.csv", "bytes": 3}, tmp_path)
    entries = json.loads((tmp_path / "DOWNLOAD_LOG.json").read_text())
    assert sorted((e["file"], e["bytes"]) for e in entries) == [("a.csv", 3), ("b.csv", 2)]
