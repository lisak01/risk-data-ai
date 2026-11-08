"""One adapter for files anywhere: local disk, network share, Amazon S3, Azure, Google Cloud.

The location is a URI; fsspec picks the right driver from its prefix:

    data/usd_eur.csv                              local file or network share
    s3://bank-raw/ecb/usd_eur.csv                 Amazon S3              (pip install s3fs)
    az://raw/ecb/usd_eur.csv, abfs://...          Azure Blob / ADLS Gen2 (pip install adlfs)
    gs://bank-raw/ecb/usd_eur.csv                 Google Cloud Storage   (pip install gcsfs)
    memory://bucket/usd_eur.csv                   in-memory "bucket", used in the tests

Credentials never appear in the code: each driver uses the cloud's standard
credential chain (environment variables, managed identity, IAM role, CLI login).
The file is expected in the ECB CSV format (e.g. a copy in a landing zone).
"""

from __future__ import annotations

import fsspec
import pandas as pd

from src.adapters.base import SourceAdapter, SourceError
from src.adapters.ecb import parse_ecb_csv


class FileAdapter(SourceAdapter):
    def __init__(self, uri: str):
        self.uri = str(uri)
        protocol = self.uri.split("://")[0] if "://" in self.uri else "local"
        self.name = f"file_{protocol}"  # e.g. file_s3, file_az, file_local

    def fetch(self, start_date: str | None = None) -> pd.DataFrame:
        try:
            with fsspec.open(self.uri, "rt", encoding="utf-8") as f:
                text = f.read()
        except ImportError as exc:  # driver for this cloud not installed
            raise SourceError(f"Missing driver for {self.uri}: {exc}") from exc
        except FileNotFoundError as exc:
            raise SourceError(f"File not found: {self.uri}") from exc
        except OSError as exc:  # network, permission, credentials
            raise SourceError(f"Cannot read {self.uri}: {exc}") from exc

        df = parse_ecb_csv(text)
        if start_date:
            df = df[df["rate_date"] >= pd.Timestamp(start_date)]
        return df.reset_index(drop=True)
