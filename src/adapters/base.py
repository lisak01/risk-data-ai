"""The common interface every data source implements.

Every adapter returns a DataFrame with exactly two columns:
    rate_date (datetime64, one row per day)
    rate      (float)

The ETL only talks to this interface. A new source - another REST API, a file in a
cloud bucket, a legacy file share, a database - gets its own small adapter class.
Quality checks, loading and the load log are reused unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class SourceError(RuntimeError):
    """Raised when a source cannot deliver data (network down, missing driver, bad format)."""


class SourceAdapter(ABC):
    """Common interface for all data sources."""

    name: str = "source"  # written to the load log, so every row can be traced to its source

    @abstractmethod
    def fetch(self, start_date: str | None = None) -> pd.DataFrame:
        """Return all observations on or after start_date (YYYY-MM-DD)."""


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({"rate_date": pd.Series(dtype="datetime64[ns]"),
                         "rate": pd.Series(dtype="float64")})
