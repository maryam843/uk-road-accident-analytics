"""Extra Power BI exports (round 3) built from the pipeline output and the executed notebooks.

Usage::

    python -m src.dashboard_exports

Writes five CSVs to ``dashboard/data/``:

* ``ksi_share_yearly.csv`` - KSI share per year, as reported and DfT-adjusted.
* ``ksi_rate_by_factor.csv`` - KSI share with Wilson 95% intervals by speed limit, urban/rural
  and light condition, 2015-2019.
* ``forecast_vs_actual.csv`` - monthly actuals 2017-2019 with the SARIMA forecast, its 80% and
  95% intervals, and the seasonal naive forecast. The SARIMA order is read from notebook 03's
  executed output and the result is checked against the test MAE printed there.
* ``data_trust.csv`` - rows per year: Kaggle copy (copied from notebook 00's executed output),
  this pipeline and RAS0101.
* ``anomalous_days.csv`` - the 10 most anomalous days printed by notebook 07, with their KSI
  counts from the pipeline.

``hex_hotspots.csv`` and ``top_links_risk.csv`` are written by notebook 08 and
``model_gains.csv`` by notebook 04. Numbers that come from a notebook are parsed from the
notebook's saved outputs, never retyped.
"""

from __future__ import annotations

import json
import logging
import re
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

from src import clean, config, evaluate

logger = logging.getLogger(__name__)

NOTEBOOKS = config.REPO_ROOT / "notebooks"
FACTOR_YEARS = (2015, 2019)
MAX_BYTES = config.DASHBOARD_MAX_BYTES


# ---------------------------------------------------------------------------
# Reading executed notebook outputs
# ---------------------------------------------------------------------------
class _TableParser(HTMLParser):
    """Collect the cells of every <table> as (tag, class, text) tuples, row by row."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[tuple[str, str, str]]]] = []
        self._cell: list[str] | None = None
        self._attrs: tuple[str, str] = ("", "")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.tables.append([])
        elif tag == "tr" and self.tables:
            self.tables[-1].append([])
        elif tag in ("th", "td") and self.tables:
            self._cell = []
            self._attrs = (tag, dict(attrs).get("class") or "")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("th", "td") and self._cell is not None:
            self.tables[-1][-1].append((*self._attrs, "".join(self._cell).strip()))
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _cell(notebook: Path, cell_id: str) -> dict:
    nb = json.loads(Path(notebook).read_text())
    cells = [c for c in nb["cells"] if c.get("id") == cell_id]
    if len(cells) != 1:
        raise KeyError(f"cell id {cell_id!r} not found in {notebook}")
    if not cells[0].get("outputs"):
        raise ValueError(f"cell {cell_id!r} in {notebook} has no saved outputs; execute the notebook")
    return cells[0]


def notebook_text(notebook: Path, cell_id: str) -> str:
    """All stream / plain-text output of one executed notebook cell."""
    parts = []
    for out in _cell(notebook, cell_id)["outputs"]:
        if "text" in out:
            parts.append("".join(out["text"]))
        elif "text/plain" in out.get("data", {}):
            parts.append("".join(out["data"]["text/plain"]))
    return "".join(parts)


def notebook_tables(notebook: Path, cell_id: str) -> list[pd.DataFrame]:
    """Every HTML table (pandas Styler or DataFrame) in one executed cell, as text frames.

    Column names are the last header row's column headings; the index is the row headings.
    Values stay as the formatted strings shown in the notebook.
    """
    frames = []
    for out in _cell(notebook, cell_id)["outputs"]:
        html = out.get("data", {}).get("text/html")
        if html is None:
            continue
        parser = _TableParser()
        parser.feed("".join(html) if isinstance(html, list) else html)
        for rows in parser.tables:
            header = [r for r in rows if r and all(t == "th" for t, _, _ in r)
                      and any("col_heading" in c for _, c, _ in r)]
            body = [r for r in rows if any(t == "td" for t, _, _ in r)]
            if not header or not body:
                continue
            columns = [txt for _, cls, txt in header[-1] if "col_heading" in cls]  # deepest level
            index = [next((txt for t, _, txt in r if t == "th"), None) for r in body]
            values = [[txt for t, _, txt in r if t == "td"] for r in body]
            frames.append(pd.DataFrame(values, index=index, columns=columns[-len(values[0]):]))
    return frames


def to_number(values: pd.Series) -> pd.Series:
    """Parse formatted numbers such as ``198,735``, ``+0.00%`` or ``60.9%`` (percent -> share)."""
    text = values.astype(str).str.strip()
    pct = text.str.endswith("%")
    num = pd.to_numeric(text.str.replace(r"[,%+]", "", regex=True), errors="coerce")
    return num.where(~pct, num / 100)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
def ksi_share_yearly(yearly: pd.DataFrame) -> pd.DataFrame:
    """KSI share of collisions per year, as reported and adjusted (DfT severity adjustment)."""
    out = pd.DataFrame({
        "year": yearly.index.astype(int),
        "ksi_share_reported": (yearly["ksi_collisions"] / yearly["collisions"]).to_numpy(),
        "ksi_share_adjusted": (yearly["ksi_collisions_adjusted"] / yearly["collisions"]).to_numpy(),
    })
    return out.round(6)


FACTORS: dict[str, str] = {
    "speed limit (mph)": "speed_limit",
    "urban or rural": "urban_or_rural_area_label",
    "light conditions": "light_conditions_label",
}


def ksi_rate_by_factor(collisions: pd.DataFrame) -> pd.DataFrame:
    """Collisions, KSI and KSI share with Wilson 95% intervals per level of each factor.

    Missing values form their own level, ``Missing``, so totals add up to all collisions.
    """
    rows = []
    for factor, col in FACTORS.items():
        level = collisions[col].astype("object")
        if col == "speed_limit":
            level = collisions[col].astype("Int64").astype("object")
        level = level.where(level.notna(), "Missing").astype(str)
        grp = collisions["ksi"].astype(int).groupby(level).agg(["size", "sum"])
        lo, hi = evaluate.wilson_interval(grp["sum"], grp["size"])
        for (name, r), low, high in zip(grp.iterrows(), lo, hi, strict=True):
            rows.append({"factor": factor, "level": name, "collisions": int(r["size"]), "ksi": int(r["sum"]),
                         "ksi_share": r["sum"] / r["size"], "wilson_low": low, "wilson_high": high})
    return pd.DataFrame(rows).round(6)


def sarima_order_from_notebook(notebook: Path = NOTEBOOKS / "03_forecasting.ipynb"
                               ) -> tuple[tuple[int, int, int], tuple[int, int, int, int], bool]:
    """The SARIMA order printed by notebook 03 (chosen once on 2005-2016)."""
    text = notebook_text(notebook, "6")
    m = re.search(r"SARIMA\((\d+), (\d+), (\d+)\)x\((\d+), (\d+), (\d+), (\d+)\), intercept=(True|False)", text)
    if m is None:
        raise ValueError("SARIMA order not found in notebook 03 output")
    g = [int(x) for x in m.groups()[:7]]
    return (g[0], g[1], g[2]), (g[3], g[4], g[5], g[6]), m.group(8) == "True"


def forecast_vs_actual(monthly: pd.Series, notebook: Path = NOTEBOOKS / "03_forecasting.ipynb") -> pd.DataFrame:
    """2017-2019 actuals with SARIMA (80% / 95% intervals) and seasonal naive forecasts.

    The model is refitted on 2005-2016 with notebook 03's fixed order. Raises if the test MAE
    of either forecast differs from the value printed in notebook 03 (2 decimals).
    """
    import pmdarima as pm

    order, seasonal_order, intercept = sarima_order_from_notebook(notebook)
    train = monthly.loc[:config.FORECAST_TRAIN_END]
    test = monthly.loc[config.FORECAST_TEST_START:config.FORECAST_TEST_END]
    model = pm.ARIMA(order=order, seasonal_order=seasonal_order, with_intercept=intercept,
                     suppress_warnings=True).fit(train)
    out = pd.DataFrame({"month": test.index.strftime("%Y-%m-%d"), "actual": test.to_numpy().astype(int)})
    for level in (0.80, 0.95):
        point, ci = model.predict(n_periods=len(test), return_conf_int=True, alpha=1 - level)
        out["sarima_forecast"] = np.asarray(point)
        out[f"sarima_{int(level * 100)}_lower"] = ci[:, 0]
        out[f"sarima_{int(level * 100)}_upper"] = ci[:, 1]
    out["seasonal_naive_forecast"] = evaluate.seasonal_naive(train, len(test)).to_numpy().astype(int)
    printed = notebook_tables(notebook, "11")[0]
    test_mae = to_number(printed.iloc[:, 3])  # columns: backtest MAE/RMSE/MAPE, test MAE/RMSE/MAPE
    for model_name, col in [("SARIMA", "sarima_forecast"), ("Seasonal naive", "seasonal_naive_forecast")]:
        mine = round(evaluate.mae(out["actual"], out[col]), 2)
        if mine != round(float(test_mae[model_name]), 2):
            raise AssertionError(f"{model_name} test MAE {mine} differs from notebook 03 ({test_mae[model_name]})")
    return out.round(3)


def data_trust(yearly: pd.DataFrame, ras: pd.DataFrame,
               notebook: Path = NOTEBOOKS / "00_kaggle_data_audit.ipynb") -> pd.DataFrame:
    """Rows per year: Kaggle copy (from notebook 00's output), this pipeline, RAS0101."""
    side = notebook_tables(notebook, "18")[0]
    kaggle = to_number(side["kaggle_rows"]).astype("Int64")
    kaggle.index = kaggle.index.astype(int)
    years = yearly.index.astype(int)
    out = pd.DataFrame({"year": years,
                        "kaggle_rows": pd.array(kaggle.reindex(years), dtype="Int64"),
                        "pipeline_rows": yearly["collisions"].astype(int).to_numpy(),
                        "ras0101_collisions": ras["collisions"].reindex(years).astype(int).to_numpy()})
    out["kaggle_minus_ras0101"] = out["kaggle_rows"] - out["ras0101_collisions"]
    out["pipeline_minus_ras0101"] = out["pipeline_rows"] - out["ras0101_collisions"]
    return out


def anomalous_days(collisions: pd.DataFrame,
                   notebook: Path = NOTEBOOKS / "07_anomalies.ipynb") -> pd.DataFrame:
    """Top 10 anomalous days from notebook 07's output, with KSI counts from the pipeline.

    Raises if the pipeline's collision count or KSI share for a day disagrees with the
    notebook's printed values.
    """
    top = notebook_tables(notebook, "24")[0]
    out = pd.DataFrame({"rank": np.arange(1, len(top) + 1), "date": top.index, "weekday": top["weekday"].to_numpy()})
    day = collisions.dropna(subset=["date"]).groupby(collisions["date"].dt.strftime("%Y-%m-%d"))["ksi"]
    counts = day.agg(["size", "sum", "count"]).reindex(out["date"])  # as notebook 07: mean over non-missing
    out["collisions"] = counts["size"].astype(int).to_numpy()
    out["ksi"] = counts["sum"].astype(int).to_numpy()
    out["ksi_share"] = (counts["sum"] / counts["count"]).to_numpy()
    if not np.array_equal(out["collisions"].to_numpy(), to_number(top["collisions"]).to_numpy()):
        raise AssertionError("collision counts differ from notebook 07")
    if not np.allclose(out["ksi_share"].round(3), to_number(top["ksi_share"]), atol=1e-9):
        raise AssertionError("KSI shares differ from notebook 07")
    for col in ["count_ratio", "ice_snow", "snow", "fog", "dark"]:
        out[f"{col}_share" if col != "count_ratio" else col] = to_number(top[col]).to_numpy()
    out["weekend"] = to_number(top["weekend"]).astype(int).to_numpy()
    out["most_unusual_feature"] = top["most_unusual"].to_numpy()
    return out.round(6)


def write_csv(frame: pd.DataFrame, name: str, out_dir: Path | None = None) -> Path:
    """Write ``name`` to the dashboard folder and check it stays under the size limit."""
    out_dir = out_dir or config.DASHBOARD_DATA_DIR
    path = out_dir / name
    frame.to_csv(path, index=False)
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"{name} is {size:,} bytes, over the {MAX_BYTES:,}-byte limit")
    logger.info("Wrote %s (%d bytes)", config.display_path(path), size)
    return path


def run(data_dir: Path | None = None) -> dict[str, int]:
    """Build every export handled by this module; return ``{file name: bytes}``."""
    data_dir = Path(data_dir) if data_dir else config.get_data_dir()
    processed, raw = data_dir / "processed", data_dir / "raw"
    yearly = pd.read_parquet(processed / config.YEARLY_PARQUET)
    ras = clean.load_ras0101(raw / config.RAS0101_RAW_FILE)
    monthly = pd.read_parquet(processed / config.MONTHLY_PARQUET)["collisions"].astype(float)
    monthly.index = pd.DatetimeIndex(monthly.index, freq="MS")
    coll = pd.read_parquet(processed / config.COLLISIONS_PARQUET,
                           columns=["collision_year", "date", "ksi", *FACTORS.values()])
    recent = coll[coll["collision_year"].between(*FACTOR_YEARS) & coll["ksi"].notna()]
    exports = {
        "ksi_share_yearly.csv": ksi_share_yearly(yearly),
        "ksi_rate_by_factor.csv": ksi_rate_by_factor(recent),
        "forecast_vs_actual.csv": forecast_vs_actual(monthly),
        "data_trust.csv": data_trust(yearly, ras),
        "anomalous_days.csv": anomalous_days(coll),
    }
    return {name: write_csv(frame, name).stat().st_size for name, frame in exports.items()}


def main() -> None:
    """CLI entry point: ``python -m src.dashboard_exports``."""
    config.setup_logging()
    for name, size in run().items():
        print(f"{name}: {size:,} bytes")


if __name__ == "__main__":
    main()
