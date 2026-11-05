"""ETL for the ECB USD/EUR reference rate into a local DuckDB file.

Usage:
    python -m src.etl --initial      # full history, table rebuilt
    python -m src.etl --daily        # only new days, idempotent upsert
    python -m src.etl --daily --source-file some.csv   # offline: same flow, file source

Flow of one run:
    1. extract   adapter.fetch(start_date)             -> DataFrame(rate_date, rate)
    2. check     hard checks on the new batch          -> stop and log FAILED if broken
    3. load      one transaction: rebuild or upsert    -> fx_rates
    4. check     soft checks on the full stored series -> warnings (gaps, jumps)
    5. log       one row per run                       -> etl_load_log
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
from dateutil.easter import easter

import config
from src.ecb_client import CsvFileAdapter, EcbClient, SourceAdapter, SourceError

# --- Database schema ---------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS fx_rates (
    series_id  VARCHAR   NOT NULL,   -- e.g. EXR.D.USD.EUR.SP00.A
    rate_date  DATE      NOT NULL,   -- ECB reference date
    rate       DOUBLE    NOT NULL,   -- USD per 1 EUR
    source     VARCHAR   NOT NULL,   -- adapter that delivered the value
    loaded_at  TIMESTAMP NOT NULL,   -- when this row was written (UTC)
    PRIMARY KEY (series_id, rate_date)  -- guarantees: one rate per day
);

CREATE TABLE IF NOT EXISTS etl_load_log (
    run_at       TIMESTAMP NOT NULL,  -- UTC
    mode         VARCHAR   NOT NULL,  -- initial | daily
    series_id    VARCHAR   NOT NULL,
    source       VARCHAR   NOT NULL,
    rows_fetched INTEGER,
    rows_new     INTEGER,             -- dates that did not exist before
    min_date     DATE,                -- date range of the fetched batch
    max_date     DATE,
    status       VARCHAR   NOT NULL,  -- OK | OK_WITH_WARNINGS | NO_NEW_DATA | FAILED
    checks       VARCHAR,             -- JSON with the result of every check
    message      VARCHAR
);
"""


def connect(db_path: Path = config.DUCKDB_PATH) -> duckdb.DuckDBPyConnection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA_SQL)
    return con


# --- Quality checks ----------------------------------------------------------

def check_batch(df: pd.DataFrame) -> dict:
    """Hard checks on the fetched batch. Any failure stops the load."""
    return {
        "valid_dates": int(df["rate_date"].isna().sum()) == 0,
        "valid_values": int(df["rate"].isna().sum()) == 0,
        "rate_positive": bool((df["rate"].dropna() > 0).all()),
        "no_duplicate_dates": not df["rate_date"].duplicated().any(),
    }


def target_holidays(years) -> set[pd.Timestamp]:
    """TARGET2 closing days: the ECB publishes no reference rate on these days.

    New Year, Good Friday, Easter Monday, 1 May, 25 and 26 December.
    """
    days = set()
    for y in years:
        e = pd.Timestamp(easter(y))
        days |= {
            pd.Timestamp(y, 1, 1), e - pd.Timedelta(days=2), e + pd.Timedelta(days=1),
            pd.Timestamp(y, 5, 1), pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 26),
        }
    return days


def check_series(dates: pd.Series, rates: pd.Series) -> dict:
    """Soft checks on the full stored series. Results are warnings, not errors.

    Gaps: ECB publishes on TARGET business days only, so weekends and TARGET
    holidays are expected gaps. Any other missing weekday is reported.
    """
    dates = pd.to_datetime(dates).sort_values()
    if dates.empty:
        return {"unexplained_missing_weekdays": [], "weekend_dates": [], "large_jumps": []}

    weekdays = pd.bdate_range(dates.min(), dates.max())  # Mon-Fri
    holidays = target_holidays(range(dates.min().year, dates.max().year + 1))
    expected = {d for d in weekdays if d not in holidays}
    missing = sorted(expected - set(dates))
    weekend = sorted(d for d in dates if d.weekday() >= 5)

    s = pd.Series(rates.values, index=dates.values).sort_index()
    change = s.pct_change().abs()
    jumps = change[change > config.MAX_ABS_DAILY_CHANGE]

    fmt = "%Y-%m-%d"
    return {
        "unexplained_missing_weekdays": [d.strftime(fmt) for d in missing],
        "weekend_dates": [d.strftime(fmt) for d in weekend],
        "large_jumps": [pd.Timestamp(d).strftime(fmt) for d in jumps.index],
    }


# --- Main run ----------------------------------------------------------------

def run(mode: str, adapter: SourceAdapter, db_path: Path = config.DUCKDB_PATH,
        series_id: str = config.SERIES_ID) -> dict:
    """Execute one ETL run and return the log entry as a dict."""
    run_at = datetime.now(UTC).replace(tzinfo=None)
    con = connect(db_path)
    log = {"run_at": run_at, "mode": mode, "series_id": series_id, "source": adapter.name,
           "rows_fetched": None, "rows_new": None, "min_date": None, "max_date": None,
           "status": "FAILED", "checks": {}, "message": ""}
    in_tx = False  # True while the load transaction is open
    try:
        # 1. Decide which period to request.
        if mode == "initial":
            start = config.INITIAL_START_DATE
        elif mode == "daily":
            last = con.execute("SELECT max(rate_date) FROM fx_rates WHERE series_id = ?",
                               [series_id]).fetchall()[0][0]
            if last is None:
                raise RuntimeError("No data stored yet. Run 'python -m src.etl --initial' first.")
            start = (last - timedelta(days=config.DAILY_LOOKBACK_DAYS)).isoformat()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 2. Extract.
        batch = adapter.fetch(start)
        log["rows_fetched"] = len(batch)
        if not batch.empty:
            log["min_date"] = batch["rate_date"].min()
            log["max_date"] = batch["rate_date"].max()

        if batch.empty:
            if mode == "initial":
                raise RuntimeError("Source returned no data for the initial load.")
            log.update(status="NO_NEW_DATA", rows_new=0,
                       message=f"No observations since {start} (not yet published?).")
            return log

        # 3. Hard checks on the batch.
        hard = check_batch(batch)
        log["checks"]["batch"] = hard
        if not all(hard.values()):
            failed = [k for k, ok in hard.items() if not ok]
            raise RuntimeError(f"Quality checks failed, nothing loaded: {failed}")

        # 4. Load in one transaction: if anything fails, the old data stays untouched.
        rows = batch.assign(series_id=series_id, source=adapter.name, loaded_at=run_at)
        con.register("new_rows", rows)
        con.execute("BEGIN TRANSACTION")
        in_tx = True
        if mode == "initial":
            con.execute("DELETE FROM fx_rates WHERE series_id = ?", [series_id])
        rows_new = con.execute(
            "SELECT count(*) FROM new_rows n WHERE NOT EXISTS ("
            " SELECT 1 FROM fx_rates f"
            " WHERE f.series_id = n.series_id AND f.rate_date = n.rate_date)"
        ).fetchall()[0][0]
        # Upsert on (series_id, rate_date): a second run with the same data
        # overwrites the same rows instead of creating duplicates.
        con.execute("""
            INSERT INTO fx_rates (series_id, rate_date, rate, source, loaded_at)
            SELECT series_id, CAST(rate_date AS DATE), rate, source, loaded_at FROM new_rows
            ON CONFLICT (series_id, rate_date) DO UPDATE SET
                rate = excluded.rate, source = excluded.source, loaded_at = excluded.loaded_at
        """)
        con.execute("COMMIT")
        in_tx = False
        con.unregister("new_rows")
        log["rows_new"] = int(rows_new)

        # 5. Soft checks on the full stored series.
        stored = con.execute("SELECT rate_date, rate FROM fx_rates WHERE series_id = ? "
                             "ORDER BY rate_date", [series_id]).df()
        soft = check_series(stored["rate_date"], stored["rate"])
        log["checks"]["series"] = {k: {"count": len(v), "dates": v[:20]} for k, v in soft.items()}
        has_warnings = any(soft.values())
        log["status"] = "OK_WITH_WARNINGS" if has_warnings else "OK"
        log["message"] = (f"{rows_new} new rows, {len(stored)} rows stored, "
                          f"last date {stored['rate_date'].max():%Y-%m-%d}.")
        return log

    except (SourceError, RuntimeError, ValueError, duckdb.Error) as exc:
        if in_tx:
            con.execute("ROLLBACK")
        log["status"] = "FAILED"
        log["message"] = str(exc)
        return log

    finally:
        _write_log(con, log)
        con.close()


def _write_log(con: duckdb.DuckDBPyConnection, log: dict) -> None:
    con.execute(
        "INSERT INTO etl_load_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [log["run_at"], log["mode"], log["series_id"], log["source"], log["rows_fetched"],
         log["rows_new"],
         log["min_date"].date() if log["min_date"] is not None else None,
         log["max_date"].date() if log["max_date"] is not None else None,
         log["status"], json.dumps(log["checks"]), log["message"]],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load the ECB USD/EUR rate into DuckDB.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--initial", action="store_true", help="full history, rebuild table")
    group.add_argument("--daily", action="store_true", help="only new days (upsert)")
    parser.add_argument("--source-file", help="read an ECB-format CSV instead of the API")
    parser.add_argument("--db", default=str(config.DUCKDB_PATH), help="DuckDB file path")
    args = parser.parse_args(argv)

    adapter = CsvFileAdapter(args.source_file) if args.source_file else EcbClient()
    log = run("initial" if args.initial else "daily", adapter, Path(args.db))

    print(f"[{log['status']}] mode={log['mode']} source={log['source']} "
          f"fetched={log['rows_fetched']} new={log['rows_new']} :: {log['message']}")
    series_checks = log["checks"].get("series", {})
    for name, result in series_checks.items():
        if result["count"]:
            print(f"  warning {name}: {result['count']} (first: {result['dates'][:5]})")
    return 1 if log["status"] == "FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
