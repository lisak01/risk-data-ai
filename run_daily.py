"""One command for the whole daily workflow: ETL (daily) + dashboard.

    python run_daily.py                  # normal daily run
    python run_daily.py --source-file x  # offline run from an ECB-format CSV

If the database is still empty, the initial load runs first. Exit code 1 if the
ETL fails; the dashboard is then still rebuilt from the last good data, and its
status badge shows the problem.
"""

import argparse
import sys

import duckdb

import config
from src import dashboard, etl
from src.ecb_client import CsvFileAdapter, EcbClient


def has_data() -> bool:
    if not config.DUCKDB_PATH.exists():
        return False
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    try:
        return con.execute("SELECT count(*) FROM fx_rates").fetchall()[0][0] > 0
    except duckdb.Error:
        return False
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", help="read an ECB-format CSV instead of the API")
    args = parser.parse_args()
    adapter = CsvFileAdapter(args.source_file) if args.source_file else EcbClient()

    mode = "daily" if has_data() else "initial"
    log = etl.run(mode, adapter)
    print(f"ETL [{log['status']}] mode={mode}: {log['message']}")

    if not has_data():
        print("No data in the database - dashboard not built.")
        return 1
    print(f"Dashboard written to {dashboard.build()}")
    return 1 if log["status"] == "FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
