"""Source adapters: one interface, many sources.

    make_adapter("ecb")                          -> ECB Data Portal REST API
    make_adapter("s3://bank-raw/usd_eur.csv")    -> file in Amazon S3
    make_adapter("data/usd_eur.csv")             -> local file

The source used by the daily run is set in config.FX_SOURCE.
"""

from __future__ import annotations

import importlib.util

from src.adapters.base import SourceAdapter, SourceError
from src.adapters.ecb import EcbClient, parse_ecb_csv
from src.adapters.files import FileAdapter

__all__ = ["SourceAdapter", "SourceError", "EcbClient", "FileAdapter", "parse_ecb_csv",
           "make_adapter", "catalogue"]


def make_adapter(source: str) -> SourceAdapter:
    """Pick the adapter from a short name or a URI."""
    if source == "ecb":
        return EcbClient()
    return FileAdapter(source)


def catalogue() -> list[dict]:
    """All connectable source types, with the driver they need and whether it is installed.

    Shown on the Data Pipeline page of the portal.
    """
    def installed(module: str | None) -> bool:
        return module is None or importlib.util.find_spec(module) is not None

    rows = [
        ("ECB Data Portal", "REST API (SDMX, CSV)", "ecb", None,
         "Market data from public / vendor APIs", "Live: daily source of the FX rate"),
        ("Local file / network share", "File (CSV)", "data/usd_eur.csv", None,
         "Legacy file drops, exports from old systems", "Tested (unit tests)"),
        ("Amazon S3", "Object storage", "s3://bucket/path.csv", "s3fs",
         "Data lake on AWS", "Same code path; needs driver + credentials"),
        ("Azure Blob / ADLS Gen2", "Object storage", "az://container/path.csv", "adlfs",
         "Data lake on Azure", "Same code path; needs driver + credentials"),
        ("Google Cloud Storage", "Object storage", "gs://bucket/path.csv", "gcsfs",
         "Data lake on GCP", "Same code path; needs driver + credentials"),
        ("In-memory bucket", "Test double for a cloud bucket", "memory://bucket/path.csv", None,
         "Simulates a cloud bucket in tests", "Tested (unit tests)"),
    ]
    return [{"name": n, "kind": k, "example": e, "driver": d or "built in",
             "driver_installed": installed(d), "use": u, "status": s}
            for n, k, e, d, u, s in rows]
