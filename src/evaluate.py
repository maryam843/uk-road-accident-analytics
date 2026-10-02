"""Metrics, effect sizes and backtesting helpers shared by the notebooks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Categorical colours in fixed order (validated colour-blind-safe for adjacent series);
# sequential charts use a single hue (matplotlib "Blues").
PALETTE: list[str] = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
SEQUENTIAL_CMAP = "Blues"


def apply_style() -> None:
    """Matplotlib defaults: thin 2px lines, recessive grid, fixed colour order."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "figure.figsize": (9, 4.5),
            "axes.prop_cycle": cycler(color=PALETTE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d9d9d6",
            "grid.linewidth": 0.6,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelcolor": "#52514e",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )


def save_fig(fig: object, name: str, figures_dir: Path | None = None) -> Path:
    """Save a matplotlib figure as ``reports/figures/<name>.png`` and return the path."""
    from src import config

    out_dir = figures_dir or config.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")  # type: ignore[attr-defined]
    logger.info("Saved figure %s", path)
    return path


# ---------------------------------------------------------------------------
# Forecast metrics
# ---------------------------------------------------------------------------
def mae(y: np.ndarray | pd.Series, yhat: np.ndarray | pd.Series) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y: np.ndarray | pd.Series, yhat: np.ndarray | pd.Series) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def mape(y: np.ndarray | pd.Series, yhat: np.ndarray | pd.Series) -> float:
    """Mean absolute percentage error in percent (``y`` must be non-zero)."""
    y_arr = np.asarray(y, float)
    if np.any(y_arr == 0):
        raise ValueError("MAPE is undefined when an actual value is 0")
    return float(np.mean(np.abs((y_arr - np.asarray(yhat, float)) / y_arr)) * 100)


def forecast_metrics(y: pd.Series, yhat: pd.Series) -> dict[str, float]:
    """MAE, RMSE and MAPE for aligned actual and forecast series."""
    return {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat), "MAPE_%": mape(y, yhat)}


def seasonal_naive(train: pd.Series, horizon: int, season: int = 12) -> pd.Series:
    """Repeat the last observed season: forecast for t+h is the value at t+h-season*k."""
    if len(train) < season:
        raise ValueError("training series shorter than one season")
    last = train.iloc[-season:].to_numpy()
    values = np.resize(last, horizon)
    index = pd.date_range(train.index[-1], periods=horizon + 1, freq=train.index.freq)[1:]
    return pd.Series(values, index=index, name="seasonal_naive")


ForecastFn = Callable[[pd.Series, int], pd.Series]


def rolling_origin_backtest(
    series: pd.Series, origins: list[pd.Timestamp], horizon: int, models: dict[str, ForecastFn]
) -> pd.DataFrame:
    """Fit each model on data up to each origin and forecast ``horizon`` steps.

    Returns:
        Long frame with columns ``model, origin, date, actual, forecast``.
    """
    rows = []
    for origin in origins:
        train = series.loc[:origin]
        actual = series.loc[origin:].iloc[1 : horizon + 1]
        for name, fn in models.items():
            fc = fn(train, len(actual))
            rows.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "origin": origin,
                        "date": actual.index,
                        "actual": actual.to_numpy(),
                        "forecast": np.asarray(fc, float)[: len(actual)],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def summarise_forecasts(results: pd.DataFrame) -> pd.DataFrame:
    """MAE / RMSE / MAPE per model over every (origin, date) row of a backtest."""
    return results.groupby("model").apply(
        lambda g: pd.Series(forecast_metrics(g["actual"], g["forecast"])), include_groups=False
    )


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def choose_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Threshold that maximises KSI F1 on the given (validation) data."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1))])


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """Threshold-free and thresholded metrics for the positive (KSI) class.

    Accuracy is returned next to the majority-class baseline accuracy on the same rows,
    because accuracy alone is misleading at roughly 1 KSI in 7 collisions.
    """
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    majority = int(np.mean(y_true) >= 0.5)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "ksi_share": float(np.mean(y_true)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "ksi_precision": float(precision_score(y_true, pred, zero_division=0)),
        "ksi_recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "majority_baseline_accuracy": float(np.mean(y_true == majority)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------
def cramers_v(table: pd.DataFrame | np.ndarray) -> tuple[float, float, float, int]:
    """Chi-square test of independence with Cramér's V.

    Returns:
        ``(chi2, p_value, cramers_v, dof)``. V = sqrt(chi2 / (n * (min(r, c) - 1))).
    """
    observed = np.asarray(table, dtype=float)
    chi2, p, dof, _ = stats.chi2_contingency(observed, correction=False)
    n = observed.sum()
    k = min(observed.shape) - 1
    return float(chi2), float(p), float(np.sqrt(chi2 / (n * k))), int(dof)


def cohens_d(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
    """Cohen's d using the pooled standard deviation."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled)


def eta_squared(groups: list[np.ndarray]) -> float:
    """Eta-squared for a one-way ANOVA: between-group SS / total SS."""
    arrays = [np.asarray(g, float) for g in groups]
    allv = np.concatenate(arrays)
    grand = allv.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in arrays)
    ss_total = ((allv - grand) ** 2).sum()
    return float(ss_between / ss_total)


def describe_effect(value: float, kind: str) -> str:
    """Conventional (Cohen) label for an effect size; a rough guide, not a rule."""
    cuts = {
        "cramers_v": (0.1, 0.3, 0.5),
        "cohens_d": (0.2, 0.5, 0.8),
        "eta_squared": (0.01, 0.06, 0.14),
    }[kind]
    v = abs(value)
    if v < cuts[0]:
        return "negligible"
    if v < cuts[1]:
        return "small"
    if v < cuts[2]:
        return "medium"
    return "large"
