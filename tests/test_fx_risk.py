"""Tests for the FX risk logic: quote direction, VaR/ES on known data, backtest."""

import math

import numpy as np
import pandas as pd
import pytest
from plotly.offline import get_plotlyjs

from src import dashboard, etl, fx_risk
from src.ecb_client import CsvFileAdapter
from tests.conftest import write_csv


def test_eur_value_divides_by_usd_per_eur_rate():
    assert fx_risk.eur_value(1.25, usd=100_000) == pytest.approx(80_000)


def test_stronger_euro_is_a_loss():
    # Rate goes from 1.00 to 1.10 USD per EUR: 100,000 USD buys 90,909 EUR instead of 100,000.
    r = math.log(1.10 / 1.00)
    assert fx_risk.relative_loss(r) == pytest.approx(1 - 1 / 1.10)
    assert fx_risk.relative_loss(r) > 0
    assert fx_risk.relative_loss(-r) < 0  # weaker euro = gain


def test_historical_var_and_es_on_known_losses():
    losses = np.arange(1, 101)  # 1, 2, ..., 100
    # numpy linear interpolation: position 0.99 * 99 = 98.01 -> 99 + 0.01 * (100 - 99)
    assert fx_risk.historical_var(losses, 0.99) == pytest.approx(99.01)
    # 97.5% quantile = 97.525 -> tail = {98, 99, 100} -> mean 99
    assert fx_risk.expected_shortfall(losses, 0.975) == pytest.approx(99)


def test_parametric_var_matches_formula():
    rng = np.random.default_rng(0)
    daily = rng.normal(0, 0.01, 10_000)
    sigma5 = np.std(daily, ddof=1) * math.sqrt(5)
    expected = 1 - math.exp(-2.326348 * sigma5)
    assert fx_risk.parametric_var(daily, 0.99, 5) == pytest.approx(expected, rel=1e-5)


def test_kupiec_and_basel():
    lr, p = fx_risk.kupiec_test(250, 2, 0.01)  # 2 exceptions, 2.5 expected -> fine
    assert p > 0.05
    lr, p = fx_risk.kupiec_test(250, 12, 0.01)  # far too many
    assert p < 0.01
    assert fx_risk.kupiec_test(250, 0, 0.01)[0] == pytest.approx(-2 * 250 * math.log(0.99))
    assert [fx_risk.basel_zone(x) for x in (4, 5, 9, 10)] == ["Green", "Yellow", "Yellow", "Red"]


def synthetic_rates(n=1200, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(1.1 * np.exp(np.cumsum(rng.normal(0, 0.005, n))), index=idx)


def test_rolling_var_has_no_look_ahead():
    rates = synthetic_rates()
    base = fx_risk.rolling_risk(rates)
    shocked = rates.copy()
    shocked.iloc[-50:] *= 1.3  # change only the future
    after = fx_risk.rolling_risk(shocked)
    cutoff = rates.index[-51]
    pd.testing.assert_series_equal(base["var"][:cutoff], after["var"][:cutoff])


def test_forward_loss_is_the_next_five_days():
    rates = synthetic_rates()
    roll = fx_risk.rolling_risk(rates)
    t = roll.index[100]
    i = rates.index.get_loc(t)
    expected = 1 - rates.iloc[i] / rates.iloc[i + 5]
    assert roll.loc[t, "fwd_loss"] == pytest.approx(expected)


def test_dashboard_builds_from_database(tmp_path):
    rates = synthetic_rates()
    src = write_csv(tmp_path / "r.csv", rates.index, rates.round(4).values)
    db = tmp_path / "t.duckdb"
    etl.run("initial", CsvFileAdapter(src), db)
    out = dashboard.build(db, tmp_path / "d.html")
    page = out.read_text(encoding="utf-8")
    assert "{{" not in page.replace(get_plotlyjs(), "")  # all placeholders filled
    assert "With 99% confidence" in page
