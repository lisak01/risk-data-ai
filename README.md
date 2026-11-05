# From market data to risk insight - case study

Three tasks in one project: an ECB exchange-rate ETL (1a), a daily FX-risk
dashboard (1b) and an index correlation analysis (2).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest        # tests, no network needed
ruff check .            # lint
```

Input files for Task 2 (`dim_master.csv`, `price_hist.parquet`) go into `data/`.

## Architecture

```
 ECB Data Portal API ──┐
 (SDMX REST, CSV)      │   src/ecb_client.py          src/etl.py                  data/risk.duckdb
                       ├─> SourceAdapter.fetch() ──>  checks -> upsert -> log ──> fx_rates
 Local CSV file  ──────┘   (EcbClient,                                            etl_load_log
 (offline / legacy)         CsvFileAdapter)                                           │
                                                                                      v
                                                                        src/fx_risk.py -> output/dashboard.html
```

All parameters are in `config.py`.

## Task 1a - ECB ETL

```bash
python -m src.etl --initial     # full history since 1999, table rebuilt
python -m src.etl --daily       # only new days
python -m src.etl --daily --source-file path/to/ecb.csv   # same flow from a file
```

**Source.** Series `EXR.D.USD.EUR.SP00.A`: daily ECB euro reference rate,
quoted as **USD per 1 EUR**. Requested as CSV from
`https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata&startPeriod=...`.

**Design decisions**

| Topic | Decision | Why |
|-------|----------|-----|
| Storage | One local DuckDB file | No server, SQL access, fast, a single file to hand over |
| History | Full history from 1999 | The series is small (~7,000 rows); risk windows can be chosen later |
| Adapter | `SourceAdapter` with `fetch(start_date)` returning `(rate_date, rate)` | A new source only needs a new adapter; checks, load and log are reused |
| Initial mode | Delete + insert in one transaction | A clean rebuild; if anything fails, the old data stays |
| Daily mode | Request from last stored date minus 5 days, upsert on `(series_id, rate_date)` | Idempotent: running twice gives the same table; also picks up ECB corrections |
| Hard checks (stop the load) | valid dates, valid values, rate > 0, no duplicate dates | Bad data must never reach the risk numbers |
| Soft checks (warnings) | missing weekdays that are not TARGET holidays, weekend dates, daily moves > 5% | Worth a look, but no reason to block the dashboard |
| Errors | 30 s timeout, 3 attempts with backoff (2 s, 4 s) for timeouts, 5xx, 429; clear `FAILED` message and exit code 1 | A scheduler can alert on the exit code |
| Log | One row per run in `etl_load_log` | Full audit trail: when, which mode, how many new rows, which date range, check results |

**Gaps are normal.** The ECB publishes only on TARGET business days. No rate on
weekends, 1 January, Good Friday, Easter Monday, 1 May, 25 and 26 December. The gap
check knows these days and reports only other missing weekdays (for example early
years with different closing days).

**HTTP 404 = no data.** The API answers 404 "No results found" if the requested
period has no observation yet (e.g. a daily run before publication). The ETL logs
this as `NO_NEW_DATA`, not as an error.

**Inspect the data**

```bash
python -c "import duckdb; c=duckdb.connect('data/risk.duckdb'); print(c.sql('SELECT * FROM etl_load_log ORDER BY run_at DESC LIMIT 5')); print(c.sql('SELECT * FROM fx_rates ORDER BY rate_date DESC LIMIT 5'))"
```

**Automating the daily run.** The ECB publishes the reference rates at around
16:00 CET. The daily job runs at 16:30 CET on weekdays, with a second run at 18:00
as a safety net (safe because the load is idempotent).

Linux / macOS (cron):
```
CRON_TZ=Europe/Berlin
30 16 * * 1-5  cd /path/to/project && .venv/bin/python run_daily.py >> logs/daily.log 2>&1
0  18 * * 1-5  cd /path/to/project && .venv/bin/python run_daily.py >> logs/daily.log 2>&1
```

Windows (Task Scheduler):
```
schtasks /Create /TN "FX risk daily" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 16:30 ^
  /TR "C:\path\to\project\.venv\Scripts\python.exe C:\path\to\project\run_daily.py"
```

In a bank setup the same command would run in the central scheduler
(e.g. Airflow, Control-M) with alerting on exit code 1.

## Task 1b - FX-risk dashboard

_Next step._

## Task 2 - Index analysis

_Next step._
