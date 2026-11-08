"""One command for everything:

    python run_all.py              # load data, compute, test, build the portal, open it
    python run_all.py --no-open    # same, without opening the browser

Steps:
    1. ETL       ECB rates into DuckDB (initial load if the database is empty, else daily)
    2. Task 2    index analysis (if the supplied files are in data/)
    3. Quality   pytest (results -> output/test_results.xml) and ruff (-> output/ruff.json)
    4. Portal    output/dashboard.html with all tabs, opened in the browser
"""

import argparse
import subprocess
import sys
import webbrowser

import config
from run_daily import has_data
from src import etl, index_analysis, portal
from src.adapters import make_adapter


def run_quality_checks() -> None:
    """Run pytest and ruff as separate processes; their result files feed the Quality tab."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"--junitxml={portal.TEST_RESULTS}"],
        cwd=config.PROJECT_ROOT, capture_output=True, text=True)
    print("   " + (tests.stdout.strip().splitlines() or ["no output"])[-1])
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", ".", "--output-format", "json"],
        cwd=config.PROJECT_ROOT, capture_output=True, text=True)
    portal.RUFF_RESULTS.write_text(ruff.stdout or "[]", encoding="utf-8")
    print(f"   ruff: {'no issues' if ruff.returncode == 0 else 'issues found'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    parser.add_argument("--skip-tests", action="store_true", help="skip pytest and ruff")
    args = parser.parse_args()

    print("1/4 ETL")
    mode = "daily" if has_data() else "initial"
    log = etl.run(mode, make_adapter(config.FX_SOURCE))
    print(f"   [{log['status']}] {mode}: {log['message']}")
    if not has_data():
        print("No FX data in the database - cannot build the portal.")
        return 1

    print("2/4 Index analysis")
    index_res = None
    if config.PRICE_HIST_PATH.exists():
        index_res = index_analysis.analyse()
        index_analysis.save_outputs(index_res)
        d = index_res["daily"]
        print(f"   avg correlation normal {d['avg_normal']:.2f} -> crisis {d['avg_crisis']:.2f}")
    else:
        print(f"   skipped: {config.PRICE_HIST_PATH} not found")

    print("3/4 Tests and code check")
    if args.skip_tests:
        print("   skipped")
    else:
        run_quality_checks()

    print("4/4 Portal")
    out = portal.build(index_res=index_res, include_index=index_res is not None)
    print(f"   {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    return 1 if log["status"] == "FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
