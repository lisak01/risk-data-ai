"""FX risk of a USD position seen from EUR: VaR, ES, parametric benchmark, backtest.

Direction of the quote (the most important line in this file):
    The ECB rate S is USD per 1 EUR.  EUR value = USD amount / S.
    S goes UP   (stronger euro)  -> fewer EUR for our USD -> LOSS.
    S goes DOWN (weaker euro)    -> more EUR for our USD  -> gain.

Method (historical simulation, overlapping 5-day returns):
    r5_t   = ln(S_t / S_{t-5})               5-trading-day log return of the rate
    loss_t = 1 - exp(-r5_t)                  relative loss of the EUR value (share of position)
    VaR    = 99% quantile of loss_t over the last WINDOW_DAYS observations
    ES     = average of loss_t at or above the 97.5% quantile
    EUR    = share * current EUR value of the position

All losses are positive numbers (loss = +, gain = -).
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import duckdb
import numpy as np
import pandas as pd

import config

# --- Building blocks ---------------------------------------------------------


def load_rates(db_path: Path = config.DUCKDB_PATH, series_id: str = config.SERIES_ID) -> pd.Series:
    """Read the stored ECB rates as a date-indexed Series (USD per EUR)."""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("SELECT rate_date, rate FROM fx_rates WHERE series_id = ? ORDER BY rate_date",
                     [series_id]).df()
    con.close()
    return pd.Series(df["rate"].values, index=pd.to_datetime(df["rate_date"]), name="rate")


def eur_value(rate: float | pd.Series, usd: float = config.POSITION_USD):
    """EUR value of the USD position. Rate is USD per 1 EUR, so we divide."""
    return usd / rate


def horizon_log_returns(rates: pd.Series, horizon: int = config.HORIZON_DAYS) -> pd.Series:
    """Overlapping h-day log returns: ln(S_t / S_{t-h}). Index = end date t."""
    return np.log(rates).diff(horizon).dropna()


def relative_loss(log_return):
    """Loss as share of the EUR value. Positive log return of S = stronger euro = loss.

    EUR value after the move: V * S_old / S_new = V * exp(-r)  ->  loss = 1 - exp(-r)
    """
    return 1 - np.exp(-log_return)


def historical_var(losses, confidence: float = config.VAR_CONFIDENCE) -> float:
    """VaR = empirical quantile of the losses (numpy default: linear interpolation)."""
    return float(np.quantile(losses, confidence))


def expected_shortfall(losses, confidence: float = config.ES_CONFIDENCE) -> float:
    """ES = average loss in the tail at or beyond the VaR at the same confidence."""
    losses = np.asarray(losses)
    return float(losses[losses >= np.quantile(losses, confidence)].mean())


def parametric_var(daily_log_returns, confidence: float = config.VAR_CONFIDENCE,
                   horizon: int = config.HORIZON_DAYS) -> float:
    """Benchmark: normal distribution, mean 0, square-root-of-time scaling.

    sigma_h = sigma_1day * sqrt(h);  worst rate move = z * sigma_h;  loss = 1 - exp(-move)
    """
    sigma_h = float(np.std(daily_log_returns, ddof=1)) * math.sqrt(horizon)
    z = NormalDist().inv_cdf(confidence)
    return float(relative_loss(z * sigma_h))


def kupiec_test(n: int, exceptions: int,
                p: float = 1 - config.VAR_CONFIDENCE) -> tuple[float, float]:
    """Kupiec proportion-of-failures test. H0: exception rate = p.

    Returns (LR statistic, p-value). LR ~ chi2 with 1 degree of freedom under H0.
    p-value > 5% -> the number of exceptions is consistent with the model.
    """
    x = exceptions
    phat = x / n

    def xlogy(a, b):  # a * ln(b) with the convention 0 * ln(0) = 0
        return 0.0 if a == 0 else a * math.log(b)

    log_l0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    log_l1 = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = -2 * (log_l0 - log_l1)
    p_value = math.erfc(math.sqrt(max(lr, 0) / 2))  # chi2(1) survival function
    return lr, p_value


def basel_zone(exceptions_in_250: int) -> str:
    """Basel traffic light for a 99% VaR over 250 observations."""
    if exceptions_in_250 <= 4:
        return "Green"
    if exceptions_in_250 <= 9:
        return "Yellow"
    return "Red"


# --- Today's risk figures ----------------------------------------------------


def risk_snapshot(rates: pd.Series, window: int = config.WINDOW_DAYS) -> dict:
    """All headline numbers for the latest date, in EUR and as share of the position."""
    losses = relative_loss(horizon_log_returns(rates)).iloc[-window:]
    daily = np.log(rates).diff().dropna().iloc[-window:]
    value = float(eur_value(rates.iloc[-1]))

    var_pct = historical_var(losses)
    es_pct = expected_shortfall(losses)
    par_pct = parametric_var(daily)
    return {
        "as_of": rates.index[-1],
        "rate": float(rates.iloc[-1]),
        "value_eur": value,
        "value_eur_prev": float(eur_value(rates.iloc[-1 - config.HORIZON_DAYS])),
        "var_pct": var_pct, "var_eur": var_pct * value,
        "es_pct": es_pct, "es_eur": es_pct * value,
        "param_var_pct": par_pct, "param_var_eur": par_pct * value,
        # sqrt(5) scaling of the 1-day historical VaR, only to show the convention matters
        "sqrt5_var_pct": historical_var(relative_loss(daily)) * math.sqrt(config.HORIZON_DAYS),
        "n_scenarios": len(losses),
        "window": window,
    }


# --- Rolling history and backtest --------------------------------------------


def rolling_risk(rates: pd.Series, window: int = config.WINDOW_DAYS) -> pd.DataFrame:
    """VaR/ES for every past day, using only data available on that day (no look-ahead).

    Columns (share of position):
        var, es, param_var  - risk estimate made at the end of day t
        fwd_loss            - loss actually realised from t to t+5 (known only later)
        exception           - fwd_loss > var
    """
    h = config.HORIZON_DAYS
    loss = relative_loss(horizon_log_returns(rates))  # loss_t uses S up to day t only
    daily = np.log(rates).diff()

    df = pd.DataFrame({"loss": loss})
    roll = df["loss"].rolling(window)
    df["var"] = roll.quantile(config.VAR_CONFIDENCE)  # same interpolation as np.quantile
    df["es"] = roll.apply(expected_shortfall, raw=True)
    sigma = daily.rolling(window).std().reindex(df.index)
    df["param_var"] = relative_loss(NormalDist().inv_cdf(config.VAR_CONFIDENCE)
                                    * sigma * math.sqrt(h))
    # The loss over (t, t+5] is the loss stored at t+5. Shift it back to t to compare.
    df["fwd_loss"] = df["loss"].shift(-h)
    df["value_eur"] = eur_value(rates.reindex(df.index))
    df = df.dropna(subset=["var"])
    df["exception"] = df["fwd_loss"] > df["var"]
    return df


def backtest_summary(roll: pd.DataFrame) -> dict:
    """Count exceptions, Kupiec test (all days and non-overlapping), Basel light."""
    done = roll.dropna(subset=["fwd_loss"])  # the last 5 days have no outcome yet
    h = config.HORIZON_DAYS

    n, x = len(done), int(done["exception"].sum())
    lr, p = kupiec_test(n, x)

    # Overlapping 5-day windows share 4 of 5 days, so exceptions cluster.
    # Every 5th day gives independent windows - a fairer input for Kupiec.
    sub = done.iloc[::-1].iloc[::h].iloc[::-1]  # every 5th day, ending at the latest one
    n_sub, x_sub = len(sub), int(sub["exception"].sum())
    lr_sub, p_sub = kupiec_test(n_sub, x_sub)

    last = done.iloc[-config.BACKTEST_DAYS:]
    x_last = int(last["exception"].sum())
    return {
        "n": n, "exceptions": x, "rate": x / n, "kupiec_lr": lr, "kupiec_p": p,
        "n_nonoverlap": n_sub, "exceptions_nonoverlap": x_sub, "kupiec_p_nonoverlap": p_sub,
        "n_last": len(last), "exceptions_last": x_last,
        "kupiec_p_last": kupiec_test(len(last), x_last)[1],
        "basel_zone": basel_zone(x_last),
        "start": done.index[0], "end": done.index[-1],
    }


def save_results(roll: pd.DataFrame, db_path: Path = config.DUCKDB_PATH) -> None:
    """Store the risk history in DuckDB so every dashboard number can be traced."""
    out = roll.reset_index(names="rate_date")
    out["var_eur"] = out["var"] * out["value_eur"]
    out["es_eur"] = out["es"] * out["value_eur"]
    con = duckdb.connect(str(db_path))
    con.register("r", out)
    con.execute("CREATE OR REPLACE TABLE fx_risk_history AS SELECT * FROM r")
    con.close()
