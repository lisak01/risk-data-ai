"""Tests for the ECB adapter and the ETL (no network needed)."""

import duckdb
import pandas as pd
import pytest
import requests

from src import ecb_client, etl
from src.ecb_client import CsvFileAdapter, EcbClient, SourceError
from tests.conftest import ecb_csv, write_csv

# Ten TARGET business days in January 2024 (1 Jan is a holiday).
DATES = pd.bdate_range("2024-01-02", periods=10)
RATES = [1.09 + i * 0.001 for i in range(10)]


def rows(db):
    con = duckdb.connect(str(db))
    n, d = con.execute("SELECT count(*), count(DISTINCT rate_date) FROM fx_rates").fetchall()[0]
    con.close()
    return n, d


def test_initial_then_daily_is_idempotent(tmp_path):
    db = tmp_path / "t.duckdb"
    first = write_csv(tmp_path / "a.csv", DATES[:7], RATES[:7])
    full = write_csv(tmp_path / "b.csv", DATES, RATES)

    assert etl.run("initial", CsvFileAdapter(first), db)["status"] == "OK"
    assert rows(db) == (7, 7)

    log1 = etl.run("daily", CsvFileAdapter(full), db)
    assert log1["rows_new"] == 3
    log2 = etl.run("daily", CsvFileAdapter(full), db)  # same data again
    assert log2["rows_new"] == 0
    assert rows(db) == (10, 10)  # no duplicates


def test_initial_rebuilds_table(tmp_path):
    db = tmp_path / "t.duckdb"
    etl.run("initial", CsvFileAdapter(write_csv(tmp_path / "a.csv", DATES, RATES)), db)
    etl.run("initial", CsvFileAdapter(write_csv(tmp_path / "b.csv", DATES[:4], RATES[:4])), db)
    assert rows(db) == (4, 4)


def test_daily_without_initial_fails_clearly(tmp_path):
    log = etl.run("daily", CsvFileAdapter(write_csv(tmp_path / "a.csv", DATES, RATES)),
                  tmp_path / "t.duckdb")
    assert log["status"] == "FAILED" and "--initial" in log["message"]


@pytest.mark.parametrize("bad_rates, bad_dates, check", [
    ([1.1, -1.0, 1.1], None, "rate_positive"),
    ([1.1, 1.1, 1.1], ["2024-01-02", "2024-01-02", "2024-01-03"], "no_duplicate_dates"),
])
def test_bad_batch_is_rejected_and_old_data_kept(tmp_path, bad_rates, bad_dates, check):
    db = tmp_path / "t.duckdb"
    etl.run("initial", CsvFileAdapter(write_csv(tmp_path / "ok.csv", DATES, RATES)), db)
    dates = bad_dates or DATES[:3]
    log = etl.run("initial", CsvFileAdapter(write_csv(tmp_path / "bad.csv", dates, bad_rates)), db)
    assert log["status"] == "FAILED"
    assert log["checks"]["batch"][check] is False
    assert rows(db) == (10, 10)  # transaction not even started -> old data intact


def test_every_run_is_logged(tmp_path):
    db = tmp_path / "t.duckdb"
    src = CsvFileAdapter(write_csv(tmp_path / "a.csv", DATES, RATES))
    etl.run("initial", src, db)
    etl.run("daily", src, db)
    con = duckdb.connect(str(db))
    log = con.execute("SELECT mode, rows_new, status FROM etl_load_log ORDER BY run_at").fetchall()
    assert [r[0] for r in log] == ["initial", "daily"]
    assert log[1][1] == 0


def test_gap_check_knows_target_holidays():
    # 2024-03-29 Good Friday and 2024-04-01 Easter Monday are expected gaps; 2024-04-03 is not.
    dates = pd.Series(pd.to_datetime(["2024-03-28", "2024-04-02", "2024-04-04"]))
    result = etl.check_series(dates, pd.Series([1.08, 1.08, 1.08]))
    assert result["unexplained_missing_weekdays"] == ["2024-04-03"]


def test_parse_ecb_csv_and_missing_columns():
    df = ecb_client.parse_ecb_csv(ecb_csv(DATES[:2], RATES[:2]))
    assert list(df.columns) == ["rate_date", "rate"] and len(df) == 2
    with pytest.raises(SourceError):
        ecb_client.parse_ecb_csv("A,B\n1,2\n")


class FakeResponse:
    def __init__(self, status, text=""):
        self.status_code, self.text = status, text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_client_retries_then_succeeds(monkeypatch):
    answers = [FakeResponse(503), FakeResponse(200, ecb_csv(DATES[:3], RATES[:3]))]
    monkeypatch.setattr(ecb_client.requests, "get", lambda *a, **k: answers.pop(0))
    df = EcbClient(backoff=0).fetch("2024-01-01")
    assert len(df) == 3


def test_client_gives_clear_error_when_unreachable(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("no route")
    monkeypatch.setattr(ecb_client.requests, "get", boom)
    with pytest.raises(SourceError, match="not reachable after 3 attempts"):
        EcbClient(backoff=0).fetch()


def test_client_404_means_no_new_data(monkeypatch):
    monkeypatch.setattr(ecb_client.requests, "get", lambda *a, **k: FakeResponse(404))
    assert EcbClient(backoff=0).fetch("2030-01-01").empty
