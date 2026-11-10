"""Extra Power BI exports built from the pipeline output and small notebook result files.

Usage::

    python -m src.dashboard_exports

Writes five CSVs to ``dashboard/data/``:

* ``ksi_share_yearly.csv`` - KSI share per year, as reported and DfT-adjusted.
* ``ksi_rate_by_factor.csv`` - KSI share with Wilson 95% intervals by speed limit, urban/rural
  and light condition, 2015-2019.
* ``forecast_vs_actual.csv`` - monthly actuals 2017-2019 with the SARIMA forecast, its 80% and
  95% intervals, and the seasonal naive forecast. The SARIMA order is read from
  ``reports/results/sarima_order.json`` and the result is checked against the test MAE in
  ``reports/results/forecast_test_mae.csv`` (both written by notebook 03).
* ``data_trust.csv`` - rows per year: Kaggle copy (``reports/results/kaggle_rows_by_year.csv``,
  written by notebook 00), this pipeline and RAS0101.
* ``anomalous_days.csv`` - the 10 most anomalous days (``reports/results/anomalous_days_top10.csv``,
  written by notebook 07), with their KSI counts from the pipeline.

``hex_hotspots.csv`` and ``top_links_risk.csv`` are written by notebook 08 and
``model_gains.csv`` by notebook 04. Numbers that come from a notebook are read from its result
file, never retyped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import clean, config, evaluate

logger = logging.getLogger(__name__)

FACTOR_YEARS = (2015, 2019)
MAX_BYTES = config.DASHBOARD_MAX_BYTES


# ---------------------------------------------------------------------------
# Reading notebook result files
# ---------------------------------------------------------------------------
RESULT_FILES: dict[str, str] = {
    "sarima_order.json": "03_forecasting",
    "forecast_test_mae.csv": "03_forecasting",
    "kaggle_rows_by_year.csv": "00_kaggle_data_audit",
    "anomalous_days_top10.csv": "07_anomalies",
}


def result_path(name: str, results_dir: Path | None = None) -> Path:
    """Path of one notebook result file; raises a clear error if it has not been written."""
    path = Path(results_dir or config.RESULTS_DIR) / name
    if not path.exists():
        notebook = RESULT_FILES.get(name, "the notebook that writes it")
        raise FileNotFoundError(
            f"{config.display_path(path)} is missing; run notebooks/{notebook}.ipynb to write it")
    return path


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


def sarima_order(results_dir: Path | None = None
                 ) -> tuple[tuple[int, int, int], tuple[int, int, int, int], bool]:
    """The SARIMA order notebook 03 chose once on 2005-2016 (``sarima_order.json``)."""
    saved = json.loads(result_path("sarima_order.json", results_dir).read_text())
    order = tuple(int(x) for x in saved["order"])
    seasonal = tuple(int(x) for x in saved["seasonal_order"])
    return order, seasonal, bool(saved["with_intercept"])  # type: ignore[return-value]


def forecast_vs_actual(monthly: pd.Series, results_dir: Path | None = None) -> pd.DataFrame:
    """2017-2019 actuals with SARIMA (80% / 95% intervals) and seasonal naive forecasts.

    The model is refitted on 2005-2016 with notebook 03's fixed order. Raises if the test MAE
    of either forecast differs from notebook 03's ``forecast_test_mae.csv`` (2 decimals).
    """
    import pmdarima as pm

    order, seasonal_order, intercept = sarima_order(results_dir)
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
    test_mae = pd.read_csv(result_path("forecast_test_mae.csv", results_dir), index_col="model")["test_mae"]
    for model_name, col in [("SARIMA", "sarima_forecast"), ("Seasonal naive", "seasonal_naive_forecast")]:
        mine = round(evaluate.mae(out["actual"], out[col]), 2)
        if mine != round(float(test_mae[model_name]), 2):
            raise AssertionError(f"{model_name} test MAE {mine} differs from notebook 03 ({test_mae[model_name]:.2f})")
    return out.round(3)


def data_trust(yearly: pd.DataFrame, ras: pd.DataFrame, results_dir: Path | None = None) -> pd.DataFrame:
    """Rows per year: Kaggle copy (notebook 00's ``kaggle_rows_by_year.csv``), this pipeline, RAS0101."""
    kaggle = pd.read_csv(result_path("kaggle_rows_by_year.csv", results_dir), index_col="year")["kaggle_rows"]
    kaggle = kaggle.astype("Int64")
    kaggle.index = kaggle.index.astype(int)
    years = yearly.index.astype(int)
    out = pd.DataFrame({"year": years,
                        "kaggle_rows": pd.array(kaggle.reindex(years), dtype="Int64"),
                        "pipeline_rows": yearly["collisions"].astype(int).to_numpy(),
                        "ras0101_collisions": ras["collisions"].reindex(years).astype(int).to_numpy()})
    out["kaggle_minus_ras0101"] = out["kaggle_rows"] - out["ras0101_collisions"]
    out["pipeline_minus_ras0101"] = out["pipeline_rows"] - out["ras0101_collisions"]
    return out


def anomalous_days(collisions: pd.DataFrame, results_dir: Path | None = None) -> pd.DataFrame:
    """Top 10 anomalous days from notebook 07's ``anomalous_days_top10.csv``, with KSI counts.

    Raises if the pipeline's collision count or KSI share for a day disagrees with the values
    notebook 07 saved. Ratios keep the precision notebook 07 displays (2 decimals for the count
    ratio, 3 for shares).
    """
    top = pd.read_csv(result_path("anomalous_days_top10.csv", results_dir), index_col="date")
    out = pd.DataFrame({"rank": np.arange(1, len(top) + 1), "date": top.index, "weekday": top["weekday"].to_numpy()})
    day = collisions.dropna(subset=["date"]).groupby(collisions["date"].dt.strftime("%Y-%m-%d"))["ksi"]
    counts = day.agg(["size", "sum", "count"]).reindex(out["date"])  # as notebook 07: mean over non-missing
    out["collisions"] = counts["size"].astype(int).to_numpy()
    out["ksi"] = counts["sum"].astype(int).to_numpy()
    out["ksi_share"] = (counts["sum"] / counts["count"]).to_numpy()
    if not np.array_equal(out["collisions"].to_numpy(), top["collisions"].to_numpy()):
        raise AssertionError("collision counts differ from notebook 07")
    if not np.allclose(out["ksi_share"].round(3), top["ksi_share"].round(3), atol=1e-9):
        raise AssertionError("KSI shares differ from notebook 07")
    out["count_ratio"] = top["count_ratio"].round(2).to_numpy()
    for col in ["ice_snow", "snow", "fog", "dark"]:
        out[f"{col}_share"] = top[col].round(3).to_numpy()
    out["weekend"] = top["weekend"].astype(int).to_numpy()
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
