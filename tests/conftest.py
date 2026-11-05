"""Shared test helpers: build small CSV files in the ECB 'csvdata' format."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `config` and `src` importable


def ecb_csv(dates, rates) -> str:
    """Return text that looks like the ECB API response (only relevant columns + one extra)."""
    df = pd.DataFrame({
        "KEY": "EXR.D.USD.EUR.SP00.A",
        "TIME_PERIOD": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
        "OBS_VALUE": rates,
        "OBS_STATUS": "A",
    })
    return df.to_csv(index=False)


def write_csv(path: Path, dates, rates) -> Path:
    path.write_text(ecb_csv(dates, rates), encoding="utf-8")
    return path
