"""Source adapters: fetch raw data and return it in one standard shape.

Every adapter returns a DataFrame with exactly two columns:
    rate_date (datetime64, one row per day)
    rate      (float)

The ETL only talks to this interface. A new source (another API, a legacy
file drop, a database) gets its own adapter class with a `fetch` method;
the ETL, the quality checks and the database code stay unchanged.
"""

from __future__ import annotations

import io
import time
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import requests

import config


class SourceError(RuntimeError):
    """Raised when a source cannot deliver data (network down, bad format, ...)."""


class SourceAdapter(ABC):
    """Common interface for all data sources."""

    name: str = "source"

    @abstractmethod
    def fetch(self, start_date: str | None = None) -> pd.DataFrame:
        """Return all observations on or after start_date (YYYY-MM-DD)."""


class EcbClient(SourceAdapter):
    """Adapter for the ECB Data Portal API (SDMX REST, CSV format)."""

    name = "ecb_data_portal"

    def __init__(
        self,
        flow: str = config.ECB_FLOW,
        series_key: str = config.ECB_SERIES_KEY,
        base_url: str = config.ECB_BASE_URL,
        timeout: float = config.HTTP_TIMEOUT_SECONDS,
        max_retries: int = config.HTTP_MAX_RETRIES,
        backoff: float = config.HTTP_BACKOFF_SECONDS,
    ):
        self.url = f"{base_url}/{flow}/{series_key}"
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

    def fetch(self, start_date: str | None = None) -> pd.DataFrame:
        params = {"format": "csvdata"}
        if start_date:
            params["startPeriod"] = start_date
        text = self._get_with_retry(params)
        return parse_ecb_csv(text)

    def _get_with_retry(self, params: dict) -> str:
        """GET with timeout and simple exponential backoff.

        Retries only problems that can go away by themselves
        (timeouts, connection errors, HTTP 429 and 5xx).
        """
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(self.url, params=params, timeout=self.timeout)
                # The ECB API answers 404 "No results found" when the period has no
                # observations (e.g. daily run before 16:00 CET). That is not an error.
                if resp.status_code == 404:
                    return ""
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}"
                else:
                    resp.raise_for_status()  # other 4xx: wrong request, do not retry
                    return resp.text
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except requests.HTTPError as exc:
                raise SourceError(f"ECB API rejected the request: {exc}") from exc

            if attempt < self.max_retries:
                time.sleep(self.backoff * 2 ** (attempt - 1))

        raise SourceError(
            f"ECB API not reachable after {self.max_retries} attempts "
            f"({self.url}). Last error: {last_error}"
        )


def parse_ecb_csv(text: str) -> pd.DataFrame:
    """Turn the ECB 'csvdata' response into the standard two-column shape.

    The response has many metadata columns; we only need TIME_PERIOD and OBS_VALUE.
    Dates and values are parsed here; checking them is the job of the ETL.
    """
    if not text.strip():
        return pd.DataFrame({"rate_date": pd.Series(dtype="datetime64[ns]"),
                             "rate": pd.Series(dtype="float64")})

    raw = pd.read_csv(io.StringIO(text), dtype=str)
    missing = {"TIME_PERIOD", "OBS_VALUE"} - set(raw.columns)
    if missing:
        raise SourceError(f"Unexpected ECB CSV format, missing columns: {missing}")

    return pd.DataFrame({
        # errors="coerce": invalid dates/values become NaT/NaN and are reported
        # by the quality checks instead of crashing here.
        "rate_date": pd.to_datetime(raw["TIME_PERIOD"], format="%Y-%m-%d", errors="coerce"),
        "rate": pd.to_numeric(raw["OBS_VALUE"], errors="coerce"),
    })


class CsvFileAdapter(SourceAdapter):
    """Adapter for a local CSV file in the ECB format.

    Used for offline runs and tests. It also shows how a legacy file-based
    interface plugs into the same ETL as a modern REST API.
    """

    name = "csv_file"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def fetch(self, start_date: str | None = None) -> pd.DataFrame:
        if not self.path.exists():
            raise SourceError(f"File not found: {self.path}")
        df = parse_ecb_csv(self.path.read_text(encoding="utf-8"))
        if start_date:
            df = df[df["rate_date"] >= pd.Timestamp(start_date)]
        return df.reset_index(drop=True)
