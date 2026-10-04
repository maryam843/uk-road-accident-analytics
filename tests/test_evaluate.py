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


def test_wilson_interval_matches_statsmodels() -> None:
    from statsmodels.stats.proportion import proportion_confint

    k, n = np.array([0, 3, 30, 99]), np.array([30, 40, 100, 100])
    lo, hi = evaluate.wilson_interval(k, n)
    ref_lo, ref_hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    assert np.allclose(lo, ref_lo) and np.allclose(hi, ref_hi)
    assert (lo <= k / n).all() and (hi >= k / n).all()


def test_jaccard_sets_and_flags() -> None:
    assert evaluate.jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)
    assert evaluate.jaccard(np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0])) == pytest.approx(1 / 3)
    assert np.isnan(evaluate.jaccard(set(), set()))
    with pytest.raises(ValueError):
        evaluate.jaccard(np.array([1, 0]), np.array([1, 0, 0]))


def test_weighted_average_precision_matches_sklearn() -> None:
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    score = np.round(rng.random(500) + 0.3 * y, 2)  # rounding creates ties
    w = rng.integers(0, 4, 500)
    assert evaluate.weighted_average_precision(y, score) == pytest.approx(
        average_precision_score(y, score))
    assert evaluate.weighted_average_precision(y, score, w) == pytest.approx(
        average_precision_score(y, score, sample_weight=w))


def test_bootstrap_average_precision_is_paired_and_centred() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    good, noise = y + rng.normal(0, 1, 2000), rng.random(2000)
    boot = evaluate.bootstrap_average_precision(y, {"good": good, "same": good, "noise": noise},
                                                n_boot=50, seed=3)
    assert boot.shape == (50, 3)
    assert (boot["good"] == boot["same"]).all()  # same resamples for every model
    lo, hi = evaluate.percentile_ci(boot["good"] - boot["noise"])
    assert lo > 0


def test_enrichment_ratio_and_bootstrap() -> None:
    y = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    flag = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    assert evaluate.enrichment_ratio(y, flag) == pytest.approx(1.0 / 0.3)
    ratio, lo, hi = evaluate.bootstrap_enrichment(y, flag, n_boot=200, seed=0)
    assert ratio == pytest.approx(1.0 / 0.3) and lo <= ratio <= hi
    assert np.isnan(evaluate.enrichment_ratio(y, np.zeros(10)))


def test_poisson_rate_ci_exact() -> None:
    rate, lo, hi = evaluate.poisson_rate_ci(np.array([0, 10]), np.array([1e8, 2e8]), per=1e8)
    assert rate.tolist() == pytest.approx([0.0, 5.0])
    assert lo[0] == 0 and hi[0] == pytest.approx(3.6889, abs=1e-3)  # Garwood upper for k=0
    assert lo[1] == pytest.approx(4.7954 / 2, abs=1e-3) and hi[1] == pytest.approx(18.3904 / 2, abs=1e-3)


def test_empirical_bayes_shrinks_low_exposure() -> None:
    rng = np.random.default_rng(0)
    exposure = np.r_[np.full(200, 1.0), np.full(200, 100.0)]
    true_rate = rng.gamma(shape=4, scale=0.5, size=400)  # heterogeneous true rates, mean 2
    counts = rng.poisson(true_rate * exposure)
    eb = evaluate.empirical_bayes_rates(pd.Series(counts), exposure)
    assert eb["prior_var"].iloc[0] > 0
    assert eb["weight"].iloc[:200].mean() < eb["weight"].iloc[200:].mean()
    # EB is closer to the truth than the raw rate for the low-exposure units.
    raw_err = np.abs(eb["raw_rate"].iloc[:200] - true_rate[:200]).mean()
    eb_err = np.abs(eb["eb_rate"].iloc[:200] - true_rate[:200]).mean()
    assert eb_err < raw_err
    # A homogeneous rate gives full shrinkage to the pooled mean.
    flat = evaluate.empirical_bayes_rates(np.full(50, 5), np.full(50, 2.0))
    assert np.allclose(flat["eb_rate"], 2.5) and (flat["weight"] == 0).all()


def test_empirical_bayes_groups_and_validation() -> None:
    eb = evaluate.empirical_bayes_rates(np.array([1, 1, 10, 10]), np.array([1.0, 1, 1, 1]),
                                        groups=np.array(["a", "a", "b", "b"]))
    assert eb["prior_mean"].tolist() == [1.0, 1.0, 10.0, 10.0]
    with pytest.raises(ValueError):
        evaluate.empirical_bayes_rates(np.array([1]), np.array([0.0]))
