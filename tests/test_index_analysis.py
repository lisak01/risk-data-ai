"""Tests for the index analysis: NASDAQ duplicate, outliers, correlation helpers."""

import numpy as np
import pandas as pd
import pytest

import config
from src import index_analysis as ia


def nasdaq_rows(dates, high, low):
    """Two rows per day for the shared key, in mixed order like the real file."""
    rows = []
    for i, (d, h, lo) in enumerate(zip(dates, high, low, strict=True)):
        pair = [h, lo] if i % 3 else [lo, h]
        rows += [{"asset_key": config.NASDAQ_KEY, "date": d, "close": p} for p in pair]
    return pd.DataFrame(rows)


def test_nasdaq_split_by_level_and_ambiguous_days_missing(monkeypatch):
    monkeypatch.setattr(config, "NASDAQ_ANCHORS", {})
    dates = pd.bdate_range("2020-01-01", periods=6)
    high = [120, 121, 100.5, 100.2, 130, 131]  # days 3-4: less than 1% apart
    low = [100, 101, 100.0, 100.0, 110, 111]
    out, info = ia.split_nasdaq(nasdaq_rows(dates, high, low))
    ixic = out[out.ticker == "IXIC.INDX"].set_index("date")["close"]
    ndx = out[out.ticker == "NDX.INDX"].set_index("date")["close"]
    assert list(ixic) == [120, 121, 110, 111]  # higher before, lower after the crossing
    assert list(ndx) == [100, 101, 130, 131]
    assert info["days_set_missing"] == 2


@pytest.mark.skipif(not config.PRICE_HIST_PATH.exists(), reason="input data not available")
def test_real_data_duplicate_is_detected_and_split_matches_published_closes():
    dim, px, checks = ia.load_raw()
    assert checks["duplicate_keys_in_master"][0]["asset_key"] == config.NASDAQ_KEY
    assert checks["naive_join_rows"] > checks["price_rows"]  # the naive join duplicates rows
    _, info = ia.split_nasdaq(px)
    assert info["anchors_ok"]


def test_outlier_flags_single_jump_but_not_market_wide_move():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=400)
    r = pd.DataFrame(rng.normal(0, 0.01, (400, 5)), index=idx, columns=list("ABCDE"))
    r.iloc[300, 0] = 0.15  # only A jumps -> data problem?
    r.iloc[350, :] = -0.08  # everybody falls -> real market move
    prices = 100 * np.exp(r.cumsum())
    flags = ia.flag_outliers(ia.daily_log_returns(prices), prices)
    assert list(zip(flags.ticker, flags.reason, strict=True)) == [("A", "jump")]
    assert pd.Timestamp(flags.date.iloc[0]) == idx[300]


def test_outlier_flags_return_across_data_gap():
    idx = pd.bdate_range("2015-01-01", periods=400)
    prices = pd.DataFrame({"A": np.linspace(100, 110, 400), "B": np.linspace(50, 60, 400)}, idx)
    prices.iloc[200:230, 0] = np.nan  # six weeks missing
    flags = ia.flag_outliers(ia.daily_log_returns(prices), prices)
    assert list(flags.reason) == ["gap"] and pd.Timestamp(flags.date.iloc[0]) == idx[230]


def test_average_pairwise_and_forbes_rigobon():
    corr = pd.DataFrame([[1, 0.2, 0.4], [0.2, 1, 0.6], [0.4, 0.6, 1]])
    assert ia.average_pairwise(corr) == pytest.approx(0.4)
    assert ia.forbes_rigobon(0.5, 0) == pytest.approx(0.5)  # no extra volatility -> no change
    assert ia.forbes_rigobon(0.5, 3) < 0.5  # more volatility -> lower adjusted correlation


def test_weekly_returns_are_friday_to_friday():
    idx = pd.bdate_range("2024-01-01", "2024-01-31")
    prices = pd.DataFrame({"A": np.arange(len(idx), dtype=float) + 100}, index=idx)
    w = ia.weekly_log_returns(prices)
    assert all(d.weekday() == 4 for d in w.index)
    assert w["A"].iloc[0] == pytest.approx(np.log(109 / 104))  # Fri 12 Jan vs Fri 5 Jan
