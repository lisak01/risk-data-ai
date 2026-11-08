"""The scheduled daily job: ETL (daily) + rebuild the portal. No browser, no tests.

    python run_daily.py                         # source from config.FX_SOURCE (ECB API)
    python run_daily.py --source s3://b/x.csv   # any other source, same flow

For everything at once (incl. tests, opening the page) use: python run_all.py

If the database is still empty, the initial load runs first. Exit code 1 if the
ETL fails; the dashboard is then still rebuilt from the last good data, and its
status badge shows the problem.
"""

import argparse
import sys

import duckdb

import config
from src import etl, portal
from src.adapters import make_adapter


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
    parser.add_argument("--source", default=config.FX_SOURCE,
                        help="'ecb' (API) or a file URI: local path, s3://, az://, gs://")
    args = parser.parse_args()
    adapter = make_adapter(args.source)

    mode = "daily" if has_data() else "initial"
    log = etl.run(mode, adapter)
    print(f"ETL [{log['status']}] mode={mode}: {log['message']}")

    if not has_data():
        print("No data in the database - dashboard not built.")
        return 1
    print(f"Portal written to {portal.build()}")
    return 1 if log["status"] == "FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
