"""Tests for metric, effect-size and backtest helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import evaluate


def test_forecast_metrics() -> None:
    y = np.array([100.0, 200.0])
    yhat = np.array([110.0, 180.0])
    assert evaluate.mae(y, yhat) == pytest.approx(15.0)
    assert evaluate.rmse(y, yhat) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert evaluate.mape(y, yhat) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        evaluate.mape([0.0], [1.0])


def test_seasonal_naive_repeats_last_season() -> None:
    idx = pd.date_range("2010-01-01", periods=24, freq="MS")
    s = pd.Series(np.arange(24.0), index=idx)
    fc = evaluate.seasonal_naive(s, horizon=14)
    assert fc.index[0] == pd.Timestamp("2012-01-01")
    assert fc.iloc[:12].tolist() == list(range(12, 24))
    assert fc.iloc[12:].tolist() == [12.0, 13.0]


def test_rolling_origin_backtest_never_sees_future() -> None:
    idx = pd.date_range("2010-01-01", periods=48, freq="MS")
    s = pd.Series(np.arange(48.0), index=idx)
    seen_ends = []

    def last_value(train: pd.Series, h: int) -> pd.Series:
        seen_ends.append(train.index[-1])
        return pd.Series([train.iloc[-1]] * h)

    origins = [pd.Timestamp("2011-12-01"), pd.Timestamp("2012-12-01")]
    res = evaluate.rolling_origin_backtest(s, origins, 12, {"last": last_value})
    assert seen_ends == origins
    assert len(res) == 24
    assert (res["date"] > res["origin"]).all()


def test_choose_threshold_and_metrics() -> None:
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    t = evaluate.choose_threshold(y, p)
    assert 0.3 < t <= 0.7
    m = evaluate.classification_metrics(y, p, t)
    assert m["ksi_recall"] == 1.0
    assert m["majority_baseline_accuracy"] == pytest.approx(4 / 6)
    assert m["tp"] + m["fn"] == 2


def test_effect_sizes() -> None:
    _, _, v, _ = evaluate.cramers_v(np.array([[50, 0], [0, 50]]))
    assert v == pytest.approx(1.0)
    _, _, v0, _ = evaluate.cramers_v(np.array([[25, 25], [25, 25]]))
    assert v0 == pytest.approx(0.0)
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert evaluate.cohens_d(a + 1.2909944, a) == pytest.approx(1.0, rel=1e-4)
    assert evaluate.eta_squared([np.array([1.0, 1.0]), np.array([3.0, 3.0])]) == pytest.approx(1.0)
    assert evaluate.describe_effect(0.05, "cramers_v") == "negligible"
