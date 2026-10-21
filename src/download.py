"""Download the DfT STATS19 open data files, the data guide, table RAS0101 and the AADF traffic counts.

Usage::

    python -m src.download            # all files
    python -m src.download --only guide ras0101
    python -m src.download --only aadf aadf_metadata

Downloads are streamed to ``DATA_DIR/raw`` with resume support (HTTP Range), a size
check against the server-reported length, a SHA-256 checksum and an entry in
``DATA_DIR/raw/DOWNLOAD_LOG.json``. The DfT data host answers ``HEAD`` with 405, so the
expected size is read from the headers of the ``GET`` response instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import requests
from tqdm import tqdm

from src import config

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1 << 20  # 1 MiB
TIMEOUT = (30, 300)  # connect, read seconds


def discover_link(page_url: str, pattern: str, fallback: str) -> str:
    """Return the first href on ``page_url`` matching ``pattern``, else ``fallback``.

    Args:
        page_url: gov.uk landing page to scan.
        pattern: Regular expression that the href must match.
        fallback: URL returned if the page cannot be fetched or nothing matches.
    """
    try:
        resp = requests.get(page_url, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Could not read %s (%s); using fallback link", page_url, exc)
        return fallback
    hrefs = re.findall(r'href="([^"]+)"', resp.text)
    matches = [h for h in hrefs if re.search(pattern, h, flags=re.IGNORECASE)]
    if not matches:
        logger.warning("No link matching %r on %s; using fallback link", pattern, page_url)
        return fallback
    logger.info("Discovered %s", matches[0])
    return matches[0]


def _expected_total(resp: requests.Response, already: int) -> int | None:
    """Total file size implied by a (possibly partial) GET response."""
    content_range = resp.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        return int(total) if total.isdigit() else None
    length = resp.headers.get("Content-Length")
    if length is None:
        return None
    return int(length) + (already if resp.status_code == 206 else 0)


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_download(url: str, dest: Path, max_retries: int = 4) -> dict[str, object]:
    """Stream ``url`` to ``dest`` with resume, retries and a final size check.

    A ``.part`` file holds the partial download; it is renamed only once the byte
    count matches the size the server reported.

    Returns:
        A log record (url, file, bytes, sha256, downloaded_at_utc).

    Raises:
        RuntimeError: if the size check still fails after ``max_retries`` attempts.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    expected: int | None = None

    for attempt in range(1, max_retries + 1):
        already = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={already}-"} if already else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as resp:
                if resp.status_code == 416:  # range not satisfiable: part is already complete
                    break
                resp.raise_for_status()
                if already and resp.status_code != 206:
                    logger.info("Server ignored Range header; restarting %s", dest.name)
                    already = 0
                expected = _expected_total(resp, already)
                mode = "ab" if already else "wb"
                with part.open(mode) as fh, tqdm(
                    total=expected, initial=already, unit="B", unit_scale=True, desc=dest.name
                ) as bar:
                    for block in resp.iter_content(chunk_size=CHUNK_BYTES):
                        fh.write(block)
                        bar.update(len(block))
        except requests.RequestException as exc:
            logger.warning("Attempt %d/%d for %s failed: %s", attempt, max_retries, url, exc)
            continue

        size = part.stat().st_size
        if expected is None or size == expected:
            break
        logger.warning("Size mismatch for %s: %d of %d bytes; resuming", dest.name, size, expected)
    else:
        raise RuntimeError(f"Download of {url} failed after {max_retries} attempts")

    size = part.stat().st_size
    if expected is not None and size != expected:
        raise RuntimeError(f"{dest.name}: got {size} bytes, expected {expected}")
    part.replace(dest)
    record: dict[str, object] = {
        "url": url,
        "file": dest.name,
        "bytes": size,
        "sha256": sha256_of(dest),
        "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    logger.info("Saved %s (%d bytes)", config.display_path(dest), size)
    return record


def append_log(record: dict[str, object], raw_dir: Path) -> None:
    """Add or replace the entry for ``record['file']`` in ``DOWNLOAD_LOG.json``."""
    log_path = raw_dir / config.DOWNLOAD_LOG_FILE
    entries: list[dict[str, object]] = []
    if log_path.exists():
        entries = json.loads(log_path.read_text())
    entries = [e for e in entries if e.get("file") != record["file"]]
    entries.append(record)
    log_path.write_text(json.dumps(entries, indent=2))


def targets() -> dict[str, tuple[str, str]]:
    """Map short name -> (url, local filename) for every file this project uses."""
    guide_url = discover_link(
        config.LANDING_PAGE_URL, r"data-guide.*\.xlsx$", config.DATA_GUIDE_URL_FALLBACK
    )
    ras_url = discover_link(config.RAS_TABLES_PAGE_URL, r"ras0101\.ods$", config.RAS0101_URL_FALLBACK)
    aadf_url = discover_link(
        config.ROAD_TRAFFIC_DOWNLOADS_URL, r"/dft_traffic_counts_aadf\.zip$", config.AADF_URL_FALLBACK
    )
    aadf_meta_url = discover_link(
        config.ROAD_TRAFFIC_DOWNLOADS_URL, r"all-traffic-data-metadata\.pdf$",
        config.AADF_METADATA_URL_FALLBACK,
    )
    return {
        "collision": (config.COLLISION_URL, config.COLLISION_RAW_FILE),
        "casualty": (config.CASUALTY_URL, config.CASUALTY_RAW_FILE),
        "guide": (guide_url, config.DATA_GUIDE_RAW_FILE),
        "ras0101": (ras_url, config.RAS0101_RAW_FILE),
        "aadf": (aadf_url, config.AADF_RAW_FILE),
        "aadf_metadata": (aadf_meta_url, config.AADF_METADATA_RAW_FILE),
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only", nargs="+",
        choices=["collision", "casualty", "guide", "ras0101", "aadf", "aadf_metadata"], default=None,
    )
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)
    config.setup_logging()

    raw_dir = config.get_data_dir() / "raw"
    for name, (url, filename) in targets().items():
        if args.only and name not in args.only:
            continue
        dest = raw_dir / filename
        if dest.exists() and not args.force:
            logger.info("%s already present, skipping (use --force to refresh)", dest.name)
            continue
        append_log(stream_download(url, dest), raw_dir)


if __name__ == "__main__":
    main()
